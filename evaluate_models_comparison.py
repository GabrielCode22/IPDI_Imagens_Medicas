#!/usr/bin/env python3
"""Compara ResNet18 original e filtrada no mesmo teste oficial BRISC.

Executar na raiz do projeto:
    python evaluate_models_comparison.py
    python evaluate_models_comparison.py --device cpu

O script:
- usa o mesmo ImageFolder de teste e as mesmas transformações dos dois modelos;
- não treina nem modifica checkpoints;
- calcula métricas na totalidade do teste oficial;
- faz análise secundária removendo apenas imagens envolvidas em pares pHash
  test-test detectados pela auditoria (sensibilidade, não novo teste independente);
- grava previsões pareadas por caminho para facilitar inspeção.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm

from data_pipeline import DATA_ROOT, IMAGE_SIZE, ResizeWithPadding

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results" / "model_comparison"
AUDIT_PAIRS = ROOT / "results" / "leakage_audit" / "near_duplicate_pairs.csv"
EXPECTED_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
BATCH_SIZE = 16
NUM_WORKERS = 2
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara os modelos original e filtrado no mesmo teste BRISC.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA foi solicitado, mas não está disponível.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "model_state_dict" not in obj:
        raise ValueError(f"Checkpoint inválido: {path}")
    return obj


def make_test_loader(batch_size: int, num_workers: int, device: torch.device):
    test_dir = DATA_ROOT / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Pasta de teste não encontrada: {test_dir}")
    transform = transforms.Compose([
        ResizeWithPadding(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    if dataset.classes != EXPECTED_CLASSES:
        raise ValueError(f"Ordem inesperada das classes: {dataset.classes}; esperado: {EXPECTED_CLASSES}")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    return dataset, loader


def build_model(checkpoint_path: Path, device: torch.device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    class_names = checkpoint.get("class_names", EXPECTED_CLASSES)
    if list(class_names) != EXPECTED_CLASSES:
        raise ValueError(f"Classes do checkpoint {checkpoint_path.name} incompatíveis: {class_names}")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(EXPECTED_CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def predict(model: nn.Module, loader: DataLoader, device: torch.device, label: str):
    all_logits: list[np.ndarray] = []
    all_targets: list[int] = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc=label):
            images = images.to(device, non_blocking=(device.type == "cuda"))
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
            all_logits.append(logits.float().cpu().numpy())
            all_targets.extend(targets.numpy().tolist())
    return np.concatenate(all_logits, axis=0), np.asarray(all_targets, dtype=np.int64)


def ece_top_label(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    result = 0.0
    for i in range(n_bins):
        # Intervalos [a,b), exceto o último [a,b].
        if i == n_bins - 1:
            mask = (conf >= edges[i]) & (conf <= edges[i + 1])
        else:
            mask = (conf >= edges[i]) & (conf < edges[i + 1])
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(result)


def calculate_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float | int]:
    pred = probs.argmax(axis=1)
    one_hot = np.eye(len(EXPECTED_CLASSES), dtype=np.float64)[y_true]
    eps = 1e-12
    clipped = np.clip(probs, eps, 1.0)
    return {
        "n_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "nll": float(-np.log(clipped[np.arange(len(y_true)), y_true]).mean()),
        "multiclass_brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "top_label_ece_15_bins": ece_top_label(y_true, probs, n_bins=15),
        "mean_top_confidence": float(probs.max(axis=1).mean()),
    }


def get_test_internal_phash_paths() -> set[str]:
    if not AUDIT_PAIRS.is_file():
        print("AVISO: near_duplicate_pairs.csv não encontrado; análise secundária pHash será omitida.")
        return set()
    pairs = pd.read_csv(AUDIT_PAIRS)
    required = {"path_a", "split_a", "path_b", "split_b"}
    if not required.issubset(pairs.columns):
        print("AVISO: CSV de pares pHash sem colunas esperadas; análise secundária será omitida.")
        return set()
    internal = pairs[(pairs["split_a"] == "test") & (pairs["split_b"] == "test")]
    paths = set(internal["path_a"].astype(str)) | set(internal["path_b"].astype(str))
    return {str(Path(p).expanduser().resolve()) for p in paths}


def save_confusion(y_true: np.ndarray, probs: np.ndarray, title: str, output: Path) -> None:
    matrix = confusion_matrix(y_true, probs.argmax(axis=1), labels=list(range(4)))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay(matrix, display_labels=EXPECTED_CLASSES).plot(
        ax=ax, cmap="Blues", values_format="d", xticks_rotation=30, colorbar=False
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch-size deve ser positivo e num-workers não pode ser negativo.")
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print("\n=== COMPARAÇÃO RESNET18 ORIGINAL × FILTRADA ===")
    print("Dispositivo:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Teste: conjunto oficial completo, mesma ordem e mesmas transformações para ambos os modelos.")

    dataset, loader = make_test_loader(args.batch_size, args.num_workers, device)
    paths = [str(Path(p).resolve()) for p, _ in dataset.samples]
    y_dataset = np.asarray(dataset.targets, dtype=np.int64)
    print(f"Imagens de teste: {len(dataset)} | classes: {dataset.classes}")

    checkpoints = {
        "original": CHECKPOINTS / "best_resnet18.pth",
        "filtered": CHECKPOINTS / "best_resnet18_filtered.pth",
    }
    outputs: dict[str, dict[str, Any]] = {}
    for name, checkpoint_path in checkpoints.items():
        model, checkpoint = build_model(checkpoint_path, device)
        logits, y_true = predict(model, loader, device, f"Modelo {name}")
        if not np.array_equal(y_true, y_dataset):
            raise RuntimeError("A ordem dos rótulos mudou durante a avaliação.")
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        metrics = calculate_metrics(y_true, probs)
        outputs[name] = {
            "logits": logits,
            "probs": probs,
            "metrics": metrics,
            "checkpoint": checkpoint,
        }
        print(f"\n{name.upper()}: época selecionada={checkpoint.get('best_epoch', 'n/d')} | "
              f"acurácia validação/checkpoint={checkpoint.get('best_val_loss', 'n/d')}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.6f}" if isinstance(value, float) else f"  {metric}: {value}")
        class_report = classification_report(
            y_true, probs.argmax(axis=1), labels=list(range(4)),
            target_names=EXPECTED_CLASSES, digits=4, zero_division=0,
        )
        (RESULTS / f"{name}_classification_report.txt").write_text(class_report, encoding="utf-8")
        save_confusion(y_true, probs, f"Matriz de confusão — modelo {name}", RESULTS / f"{name}_confusion_matrix.png")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Comparação pareada por mesma imagem.
    original = outputs["original"]["probs"]
    filtered = outputs["filtered"]["probs"]
    pred_original = original.argmax(axis=1)
    pred_filtered = filtered.argmax(axis=1)
    rows = []
    for i, path in enumerate(paths):
        row = {
            "image_path": path,
            "true_class": EXPECTED_CLASSES[int(y_dataset[i])],
            "original_prediction": EXPECTED_CLASSES[int(pred_original[i])],
            "filtered_prediction": EXPECTED_CLASSES[int(pred_filtered[i])],
            "original_correct": bool(pred_original[i] == y_dataset[i]),
            "filtered_correct": bool(pred_filtered[i] == y_dataset[i]),
            "prediction_changed": bool(pred_original[i] != pred_filtered[i]),
            "original_confidence": float(original[i, pred_original[i]]),
            "filtered_confidence": float(filtered[i, pred_filtered[i]]),
        }
        for c, cname in enumerate(EXPECTED_CLASSES):
            row[f"original_prob_{cname}"] = float(original[i, c])
            row[f"filtered_prob_{cname}"] = float(filtered[i, c])
        rows.append(row)
    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(RESULTS / "paired_test_predictions.csv", index=False)

    # Avaliação secundária: remove apenas as imagens explicitamente envolvidas
    # em pares test-test da triagem pHash. Não é novo teste independente.
    flagged_paths = get_test_internal_phash_paths()
    path_to_index = {p: i for i, p in enumerate(paths)}
    flagged_indices = {path_to_index[p] for p in flagged_paths if p in path_to_index}
    keep_mask = np.asarray([i not in flagged_indices for i in range(len(paths))], dtype=bool)
    secondary: dict[str, Any] = {
        "description": "Análise de sensibilidade removendo imagens diretamente envolvidas em pares test-test pHash; não constitui conjunto independente nem prova de independência por paciente.",
        "flagged_test_images_found": int(len(flagged_indices)),
        "remaining_images": int(keep_mask.sum()),
    }
    if flagged_indices:
        for name in ("original", "filtered"):
            secondary[name] = calculate_metrics(y_dataset[keep_mask], outputs[name]["probs"][keep_mask])
    else:
        secondary["note"] = "Nenhuma imagem test-test foi identificada no CSV da auditoria; análise secundária não executada."

    summary = {
        "test_protocol": "Same official BRISC classification test; same preprocessing; no weight updates or calibration fitting.",
        "device": str(device),
        "classes": EXPECTED_CLASSES,
        "original_checkpoint": str(checkpoints["original"].relative_to(ROOT)),
        "filtered_checkpoint": str(checkpoints["filtered"].relative_to(ROOT)),
        "original_best_epoch": outputs["original"]["checkpoint"].get("best_epoch"),
        "filtered_best_epoch": outputs["filtered"]["checkpoint"].get("best_epoch"),
        "full_test": {name: outputs[name]["metrics"] for name in ("original", "filtered")},
        "secondary_test_internal_phash_sensitivity": secondary,
        "n_prediction_changes": int((pred_original != pred_filtered).sum()),
        "n_both_correct": int(((pred_original == y_dataset) & (pred_filtered == y_dataset)).sum()),
        "n_original_only_correct": int(((pred_original == y_dataset) & (pred_filtered != y_dataset)).sum()),
        "n_filtered_only_correct": int(((pred_original != y_dataset) & (pred_filtered == y_dataset)).sum()),
        "caveats": [
            "pHash identifies visual-similarity candidates, not patient identity.",
            "The official test contains internal pHash-related images; full-test scores may be dependent.",
            "The secondary subset is a sensitivity analysis, not a patient-independent test set.",
            "Raw softmax probabilities are compared; no temperature was fitted on the test set.",
        ],
    }
    (RESULTS / "comparison_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Gráficos separados para não misturar métricas com escalas diferentes.
    fig, ax = plt.subplots(figsize=(7.5, 5))
    names = ["Acurácia", "F1 macro"]
    keys = ["accuracy", "macro_f1"]
    x = np.arange(len(keys))
    width = 0.34
    ax.bar(x - width / 2, [outputs["original"]["metrics"][k] for k in keys], width, label="Original")
    ax.bar(x + width / 2, [outputs["filtered"]["metrics"][k] for k in keys], width, label="Filtrada")
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.05)
    ax.set_title("Classificação — teste oficial")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "model_comparison_classification.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    calibration_specs = [
        ("nll", "NLL"),
        ("multiclass_brier", "Brier multiclasses"),
        ("top_label_ece_15_bins", "ECE (15 bins)"),
    ]
    for ax, (key, title) in zip(axes, calibration_specs):
        vals = [outputs["original"]["metrics"][key], outputs["filtered"]["metrics"][key]]
        ax.bar(["Original", "Filtrada"], vals)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Métricas probabilísticas — teste oficial (menor é melhor)")
    fig.tight_layout()
    fig.savefig(RESULTS / "model_comparison_calibration.png", dpi=160)
    plt.close(fig)

    comparison_rows = []
    for metric in outputs["original"]["metrics"]:
        comparison_rows.append({
            "metric": metric,
            "original": outputs["original"]["metrics"][metric],
            "filtered": outputs["filtered"]["metrics"][metric],
        })
    pd.DataFrame(comparison_rows).to_csv(RESULTS / "model_comparison_metrics.csv", index=False)

    print("\n=== COMPARAÇÃO PAREADA ===")
    print("Previsões diferentes:", summary["n_prediction_changes"])
    print("Ambos corretos:", summary["n_both_correct"])
    print("Somente original correto:", summary["n_original_only_correct"])
    print("Somente filtrado correto:", summary["n_filtered_only_correct"])
    print("\n=== SENSIBILIDADE — TESTE SEM IMAGENS ENVOLVIDAS EM PARES pHash TEST-TEST ===")
    print("Imagens removidas desta análise secundária:", secondary["flagged_test_images_found"])
    print("Imagens restantes:", secondary["remaining_images"])
    for name in ("original", "filtered"):
        if name in secondary:
            print(f"{name}: accuracy={secondary[name]['accuracy']:.4f} | macro-F1={secondary[name]['macro_f1']:.4f}")

    print("\n=== ARQUIVOS GERADOS ===")
    for fname in [
        "comparison_summary.json", "paired_test_predictions.csv",
        "original_classification_report.txt", "filtered_classification_report.txt",
        "original_confusion_matrix.png", "filtered_confusion_matrix.png",
        "model_comparison_classification.png", "model_comparison_calibration.png",
        "model_comparison_metrics.csv",
    ]:
        print(RESULTS / fname)
    print("\nConcluído. Nenhum peso foi atualizado; nenhum parâmetro foi ajustado no teste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
