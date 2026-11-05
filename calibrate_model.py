#!/usr/bin/env python3
"""Calibra probabilidades da ResNet18 por temperature scaling.

A temperatura e ajustada SOMENTE na particao de validacao. O conjunto oficial
 de teste serve apenas para comparar probabilidades antes/depois, sem ajustar
 pesos, temperatura ou hiperparametros.

Execute na raiz do projeto:
    python calibrate_model.py

Arquivos gerados em results/:
    calibration_metrics.json
    calibration_report.txt
    calibration_curves.png
    test_calibrated_predictions.csv
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, log_loss
from torchvision.models import resnet18
from tqdm import tqdm

from data_pipeline import build_dataloaders


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_resnet18.pth"
RESULTS_DIR = ROOT / "results"
N_BINS = 15
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 20.0


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {path}\nExecute primeiro: python train.py"
        )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint inválido: model_state_dict não encontrado.")
    return checkpoint


def collect_logits(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna logits float32 e rótulos, preservando a ordem do DataLoader."""
    model.eval()
    logits_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    use_amp = device.type == "cuda"

    with torch.inference_mode():
        for images, targets in tqdm(loader, desc=description):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
            logits_all.append(logits.float().cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    if not logits_all:
        raise RuntimeError(f"Nenhuma amostra foi carregada em: {description}")
    return np.concatenate(logits_all), np.concatenate(targets_all).astype(int)


def fit_temperature(val_logits: np.ndarray, val_targets: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Ajusta uma única temperatura minimizando NLL na validação."""
    logits = torch.tensor(val_logits, dtype=torch.float64)
    targets = torch.tensor(val_targets, dtype=torch.long)

    def objective(log_temperature: float) -> float:
        temperature = float(np.exp(log_temperature))
        with torch.no_grad():
            loss = F.cross_entropy(logits / temperature, targets)
        return float(loss.item())

    result = minimize_scalar(
        objective,
        bounds=(np.log(MIN_TEMPERATURE), np.log(MAX_TEMPERATURE)),
        method="bounded",
        options={"xatol": 1e-7, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Não foi possível ajustar a temperatura: {result.message}")
    temperature = float(np.exp(result.x))
    return temperature, {
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "validation_nll_at_optimum": float(result.fun),
        "iterations": int(result.nfev),
    }


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[y_true]
    # Definição multiclasses: média por amostra da soma dos erros quadráticos.
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def top_label_ece(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = N_BINS
) -> tuple[float, list[dict[str, float | int]]]:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Inclui confidence=1.0 no último intervalo.
    bin_ids = np.minimum(np.digitize(confidence, edges[1:-1], right=False), n_bins - 1)
    ece = 0.0
    rows: list[dict[str, float | int]] = []
    for bin_index in range(n_bins):
        mask = bin_ids == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(confidence[mask].mean())
        avg_acc = float(correct[mask].mean())
        ece += (count / len(y_true)) * abs(avg_acc - avg_conf)
        rows.append(
            {
                "bin": bin_index + 1,
                "count": count,
                "mean_confidence": avg_conf,
                "accuracy": avg_acc,
            }
        )
    return float(ece), rows


def classwise_ovr_ece(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = N_BINS
) -> tuple[float, list[float]]:
    """ECE one-vs-rest para cada classe, depois média macro."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n_classes = probabilities.shape[1]
    values: list[float] = []
    for class_id in range(n_classes):
        p = probabilities[:, class_id]
        observed = (y_true == class_id).astype(np.float64)
        bin_ids = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
        ece = 0.0
        for bin_index in range(n_bins):
            mask = bin_ids == bin_index
            count = int(mask.sum())
            if count:
                ece += (count / len(y_true)) * abs(
                    float(observed[mask].mean()) - float(p[mask].mean())
                )
        values.append(float(ece))
    return float(np.mean(values)), values


def compute_metrics(
    y_true: np.ndarray, probabilities: np.ndarray
) -> tuple[dict[str, float], list[dict[str, float | int]]]:
    predictions = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    ece, bins = top_label_ece(y_true, probabilities)
    classwise_ece, _ = classwise_ovr_ece(y_true, probabilities)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "nll": float(log_loss(y_true, probabilities, labels=list(range(probabilities.shape[1])))),
        "multiclass_brier": multiclass_brier(y_true, probabilities),
        "top_label_ece_15_bins": ece,
        "macro_classwise_ovr_ece_15_bins": classwise_ece,
        "mean_top_confidence": float(confidence.mean()),
        "accuracy_confidence_gap": float(confidence.mean() - (predictions == y_true).mean()),
    }
    return metrics, bins


def plot_reliability(
    test_before: np.ndarray,
    test_after: np.ndarray,
    test_targets: np.ndarray,
    class_names: list[str],
    output_path: Path,
) -> None:
    ece_before, bins_before = top_label_ece(test_targets, test_before)
    ece_after, bins_after = top_label_ece(test_targets, test_after)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibração perfeita")
    for bins, label, marker in [
        (bins_before, f"Antes (ECE={ece_before:.4f})", "o"),
        (bins_after, f"Depois (ECE={ece_after:.4f})", "s"),
    ]:
        if bins:
            ax.plot(
                [row["mean_confidence"] for row in bins],
                [row["accuracy"] for row in bins],
                marker=marker,
                linewidth=1.8,
                label=label,
            )
    ax.set_title("Calibração da confiança da previsão")
    ax.set_xlabel("Confiança média")
    ax.set_ylabel("Acurácia observada")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1]
    x = np.arange(len(class_names))
    width = 0.36
    _, before_per_class = classwise_ovr_ece(test_targets, test_before)
    _, after_per_class = classwise_ovr_ece(test_targets, test_after)
    ax.bar(x - width / 2, before_per_class, width, label="Antes")
    ax.bar(x + width / 2, after_per_class, width, label="Depois")
    ax.set_xticks(x, class_names, rotation=20, ha="right")
    ax.set_ylabel("ECE one-vs-rest (menor é melhor)")
    ax.set_title("Erro de calibração por classe")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.suptitle("Temperature scaling — teste oficial BRISC", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== CALIBRAÇÃO DE PROBABILIDADES — RESNET18 / BRISC ===")
    print("Dispositivo:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Método: temperature scaling (uma temperatura escalar)")
    print("Ajuste da temperatura: somente validação")

    _train_loader, val_loader, test_loader, pipeline_classes = build_dataloaders()
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    class_names = checkpoint.get("class_names", pipeline_classes)
    if class_names != pipeline_classes:
        raise ValueError(
            "A ordem das classes no checkpoint difere do pipeline.\n"
            f"Checkpoint: {class_names}\nPipeline: {pipeline_classes}"
        )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_logits, val_targets = collect_logits(model, val_loader, device, "Logits de validação")
    test_logits, test_targets = collect_logits(model, test_loader, device, "Logits do teste oficial")

    temperature, optimizer_info = fit_temperature(val_logits, val_targets)
    val_before = softmax(val_logits, axis=1)
    val_after = softmax(val_logits / temperature, axis=1)
    test_before = softmax(test_logits, axis=1)
    test_after = softmax(test_logits / temperature, axis=1)

    val_metrics_before, _ = compute_metrics(val_targets, val_before)
    val_metrics_after, _ = compute_metrics(val_targets, val_after)
    test_metrics_before, test_bins_before = compute_metrics(test_targets, test_before)
    test_metrics_after, test_bins_after = compute_metrics(test_targets, test_after)

    # A temperatura positiva não altera argmax: mudança de acurácia seria bug.
    if not np.array_equal(test_before.argmax(axis=1), test_after.argmax(axis=1)):
        raise RuntimeError("Temperature scaling alterou a classe prevista inesperadamente.")

    result = {
        "method": "scalar_temperature_scaling",
        "temperature": temperature,
        "temperature_fit_split": "validation",
        "validation_size": int(len(val_targets)),
        "test_size": int(len(test_targets)),
        "validation_metrics_before": val_metrics_before,
        "validation_metrics_after": val_metrics_after,
        "test_metrics_before": test_metrics_before,
        "test_metrics_after": test_metrics_after,
        "test_ece_bins_before": test_bins_before,
        "test_ece_bins_after": test_bins_after,
        "class_names": class_names,
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "ece_definition": "top-label ECE with 15 equal-width confidence bins",
        "classwise_ece_definition": "macro average of one-vs-rest ECE with 15 equal-width probability bins",
        "brier_definition": "multiclass mean over samples of the sum of squared one-hot probability errors",
        "optimizer": optimizer_info,
        "note": "Temperature fitted only on validation labels; official test labels were not used to fit temperature. Test set had been evaluated in earlier project steps.",
    }

    metrics_path = RESULTS_DIR / "calibration_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    report_path = RESULTS_DIR / "calibration_report.txt"
    lines = [
        "CALIBRAÇÃO DE PROBABILIDADES — RESNET18 / BRISC",
        f"Temperatura ajustada na validação: {temperature:.6f}",
        f"Checkpoint: {CHECKPOINT_PATH}",
        "",
        "VALIDAÇÃO (conjunto usado para ajustar a temperatura; leitura otimista)",
        "Métrica                         Antes          Depois",
    ]
    for key in val_metrics_before:
        lines.append(f"{key:32s} {val_metrics_before[key]:.6f}     {val_metrics_after[key]:.6f}")
    lines += ["", "TESTE OFICIAL (temperatura não ajustada neste conjunto)", "Métrica                         Antes          Depois"]
    for key in test_metrics_before:
        lines.append(f"{key:32s} {test_metrics_before[key]:.6f}     {test_metrics_after[key]:.6f}")
    lines += [
        "",
        "Nota: temperature scaling preserva as classes previstas (argmax),",
        "portanto acurácia e F1 devem permanecer iguais. O método altera apenas",
        "a distribuição de confiança. ECE depende do número e dos limites dos bins.",
        "A validação foi usada para ajustar a temperatura; priorize o resultado do teste.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    curves_path = RESULTS_DIR / "calibration_curves.png"
    plot_reliability(test_before, test_after, test_targets, class_names, curves_path)

    test_predictions = test_after.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    test_samples = getattr(test_loader.dataset, "samples", [])
    for index, (target, pred, probs) in enumerate(zip(test_targets, test_predictions, test_after)):
        row: dict[str, Any] = {
            "index": index,
            "image_path": test_samples[index][0] if len(test_samples) == len(test_targets) else "",
            "true_class": class_names[int(target)],
            "predicted_class": class_names[int(pred)],
            "correct": bool(target == pred),
            "confidence_after_calibration": float(probs[pred]),
        }
        for class_id, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(probs[class_id])
        rows.append(row)
    predictions_path = RESULTS_DIR / "test_calibrated_predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)

    print(f"\nTemperatura ajustada: {temperature:.6f}")
    print("\n=== TESTE OFICIAL — ANTES × DEPOIS ===")
    print(f"{'Métrica':38s} {'Antes':>12s} {'Depois':>12s}")
    for key in test_metrics_before:
        print(f"{key:38s} {test_metrics_before[key]:12.6f} {test_metrics_after[key]:12.6f}")
    print("\n=== ARQUIVOS GERADOS ===")
    print("Métricas:       ", metrics_path)
    print("Relatório:      ", report_path)
    print("Gráficos:       ", curves_path)
    print("Previsões:      ", predictions_path)
    print("\nConcluído. Nenhum peso foi atualizado; a temperatura foi ajustada apenas na validação.")


if __name__ == "__main__":
    main()
