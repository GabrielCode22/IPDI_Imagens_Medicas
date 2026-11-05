#!/usr/bin/env python3
"""Investiga desempenho por plano anatômico e ativações Grad-CAM no BRISC.

Usa o checkpoint já treinado e o conjunto oficial de teste. Não treina nem
atualiza pesos. Planos extraídos do sufixo do nome: ax=axial, co=coronal,
sa=sagittal (ex.: *_gl_ax_t1.jpg).

Execute na raiz de tumor_cnn:
    python investigate_planes_gradcam.py

Saídas em results/:
    orientation_counts.csv
    orientation_metrics.csv
    orientation_class_metrics.csv
    test_predictions_by_plane.csv
    orientation_accuracy_f1.png
    orientation_confusion_matrices.png
    gradcam_correct_by_plane.png
    gradcam_errors_true_vs_pred.png
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_resnet18.pth"
RESULTS_DIR = ROOT / "results"
BATCH_SIZE = 16
NUM_WORKERS = 2
MEAN_DEFAULT = (0.485, 0.456, 0.406)
STD_DEFAULT = (0.229, 0.224, 0.225)
PLANE_LABELS = {
    "ax": "Axial",
    "co": "Coronal",
    "sa": "Sagital",
}
PLANE_ORDER = ["ax", "co", "sa"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_plane(filename: str) -> str:
    """Extrai ax/co/sa do sufixo oficial do nome do arquivo BRISC."""
    stem = Path(filename).stem.lower()
    match = re.search(r"_(ax|co|sa)(?:_t1)?$", stem)
    return match.group(1) if match else "unknown"


def load_checkpoint(device: torch.device) -> dict[str, Any]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {CHECKPOINT_PATH}. Execute python train.py primeiro."
        )
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint inválido: model_state_dict não encontrado.")
    return checkpoint


def build_test_loader(checkpoint: dict[str, Any]):
    test_dir = DATA_ROOT / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Pasta de teste não encontrada: {test_dir}")

    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    mean = tuple(checkpoint.get("normalization_mean", MEAN_DEFAULT))
    std = tuple(checkpoint.get("normalization_std", STD_DEFAULT))
    transform = transforms.Compose(
        [
            ResizeWithPadding(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    class_names = checkpoint.get("class_names", dataset.classes)
    if dataset.classes != class_names:
        raise ValueError(
            f"Ordem de classes diferente. Dataset={dataset.classes}; checkpoint={class_names}"
        )
    if not dataset.samples:
        raise RuntimeError(f"Nenhuma imagem encontrada em {test_dir}")

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return dataset, loader, class_names, image_size, mean, std


def build_model(checkpoint: dict[str, Any], class_names: list[str], device: torch.device):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def evaluate_test_by_plane(model, dataset, loader, class_names, device) -> pd.DataFrame:
    targets: list[int] = []
    predictions: list[int] = []
    probabilities: list[list[float]] = []

    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for images, batch_targets in tqdm(loader, desc="Teste por orientação"):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
            preds = probs.argmax(dim=1)
            targets.extend(batch_targets.tolist())
            predictions.extend(preds.cpu().tolist())
            probabilities.extend(probs.cpu().tolist())

    if len(targets) != len(dataset.samples):
        raise RuntimeError("Quantidade de previsões não corresponde às imagens do teste.")

    probability_array = np.asarray(probabilities)
    rows = []
    for i, ((image_path, _folder_label), true_id, pred_id, probs) in enumerate(
        zip(dataset.samples, targets, predictions, probability_array)
    ):
        plane = parse_plane(Path(image_path).name)
        row = {
            "index": i,
            "image_path": image_path,
            "filename": Path(image_path).name,
            "plane_code": plane,
            "plane": PLANE_LABELS.get(plane, "Não identificado"),
            "true_class": class_names[true_id],
            "predicted_class": class_names[pred_id],
            "correct": bool(true_id == pred_id),
            "confidence": float(probs[pred_id]),
        }
        for class_idx, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(probs[class_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def scan_split_plane_distribution(class_names: list[str]) -> pd.DataFrame:
    """Conta imagens por split, classe e plano usando os nomes oficiais."""
    rows = []
    for split in ("train", "test"):
        for class_name in class_names:
            class_dir = DATA_ROOT / split / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                plane_code = parse_plane(path.name)
                rows.append(
                    {
                        "split": split,
                        "class": class_name,
                        "plane_code": plane_code,
                        "plane": PLANE_LABELS.get(plane_code, "Não identificado"),
                    }
                )
    if not rows:
        raise RuntimeError("Não foi possível contar imagens por plano nos diretórios train/test.")
    raw = pd.DataFrame(rows)
    counts = (
        raw.groupby(["split", "class", "plane_code", "plane"])
        .size()
        .rename("n_images")
        .reset_index()
    )
    return counts


def save_split_distribution_plot(distribution: pd.DataFrame) -> None:
    known = distribution[distribution["plane_code"].isin(PLANE_ORDER)]
    pivot = known.groupby(["split", "plane"])["n_images"].sum().unstack(fill_value=0)
    plane_labels = [PLANE_LABELS[p] for p in PLANE_ORDER if PLANE_LABELS[p] in pivot.columns]
    pivot = pivot.reindex(columns=plane_labels, fill_value=0)
    ax = pivot.plot(kind="bar", figsize=(8, 4.8))
    ax.set_title("Distribuição das imagens por plano e partição")
    ax.set_xlabel("Partição")
    ax.set_ylabel("Número de imagens")
    ax.legend(title="Plano anatômico")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(RESULTS_DIR / "plane_distribution_by_split.png", dpi=160)
    plt.close(ax.figure)


def make_orientation_tables(df: pd.DataFrame, class_names: list[str]):
    labels = list(range(len(class_names)))
    code_to_id = {name: idx for idx, name in enumerate(class_names)}
    summary_rows = []
    class_rows = []
    count_rows = []

    for plane_code in PLANE_ORDER + (["unknown"] if (df["plane_code"] == "unknown").any() else []):
        subset = df[df["plane_code"] == plane_code]
        if subset.empty:
            continue
        y_true = subset["true_class"].map(code_to_id).to_numpy()
        y_pred = subset["predicted_class"].map(code_to_id).to_numpy()
        summary_rows.append(
            {
                "plane_code": plane_code,
                "plane": PLANE_LABELS.get(plane_code, "Não identificado"),
                "n_images": len(subset),
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_precision": precision_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                ),
                "macro_recall": recall_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                ),
                "macro_f1": f1_score(
                    y_true, y_pred, labels=labels, average="macro", zero_division=0
                ),
                "errors": int((y_true != y_pred).sum()),
            }
        )
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        for class_name in class_names:
            item = report[class_name]
            class_rows.append(
                {
                    "plane_code": plane_code,
                    "plane": PLANE_LABELS.get(plane_code, "Não identificado"),
                    "class": class_name,
                    "support": int(item["support"]),
                    "precision": float(item["precision"]),
                    "recall": float(item["recall"]),
                    "f1_score": float(item["f1-score"]),
                }
            )
            count_rows.append(
                {
                    "plane_code": plane_code,
                    "plane": PLANE_LABELS.get(plane_code, "Não identificado"),
                    "class": class_name,
                    "n_images": int((subset["true_class"] == class_name).sum()),
                    "correct": int(
                        ((subset["true_class"] == class_name) & subset["correct"]).sum()
                    ),
                    "errors": int(
                        ((subset["true_class"] == class_name) & ~subset["correct"]).sum()
                    ),
                }
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(class_rows),
        pd.DataFrame(count_rows),
    )


def save_orientation_plots(df, summary, class_names):
    # Gráfico de volume de imagens por plano e desempenho por plano.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    count_by_plane = df["plane"].value_counts()
    display_planes = [PLANE_LABELS[p] for p in PLANE_ORDER if p in set(df["plane_code"])]
    axes[0].bar(display_planes, [int(count_by_plane.get(p, 0)) for p in display_planes])
    axes[0].set_title("Quantidade de imagens por plano")
    axes[0].set_ylabel("Número de imagens")
    axes[0].tick_params(axis="x", rotation=15)

    known = summary[summary["plane_code"].isin(PLANE_ORDER)]
    x = np.arange(len(known))
    width = 0.36
    axes[1].bar(x - width / 2, known["accuracy"], width, label="Acurácia")
    axes[1].bar(x + width / 2, known["macro_f1"], width, label="F1 macro")
    axes[1].set_xticks(x, known["plane"], rotation=15)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Métrica")
    axes[1].set_title("Desempenho por plano anatômico")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "orientation_accuracy_f1.png", dpi=160)
    plt.close(fig)

    # Matrizes de confusão independentes por plano, com as mesmas classes.
    known_codes = [p for p in PLANE_ORDER if (df["plane_code"] == p).any()]
    if not known_codes:
        return
    fig, axes = plt.subplots(1, len(known_codes), figsize=(6 * len(known_codes), 5.2), squeeze=False)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for ax, plane_code in zip(axes[0], known_codes):
        subset = df[df["plane_code"] == plane_code]
        y_true = subset["true_class"].map(class_to_idx).to_numpy()
        y_pred = subset["predicted_class"].map(class_to_idx).to_numpy()
        matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
            ax=ax, cmap="Blues", values_format="d", colorbar=False, xticks_rotation=35
        )
        ax.set_title(f"{PLANE_LABELS[plane_code]} (n={len(subset)})")
        ax.set_xlabel("Prevista")
        ax.set_ylabel("Real")
    fig.suptitle("Matrizes de confusão por plano anatômico", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RESULTS_DIR / "orientation_confusion_matrices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def prepare_image(path: Path, image_size: int, mean, std, device):
    with Image.open(path) as source:
        image = ResizeWithPadding(image_size)(source)
    display = np.asarray(image, dtype=np.float32) / 255.0
    tensor = transforms.ToTensor()(image)
    tensor = transforms.Normalize(mean, std)(tensor).unsqueeze(0).to(device)
    return tensor, display


def compute_gradcam(model, input_tensor, target_class: int):
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    layer = model.layer4[-1]
    handle = layer.register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            score = logits[0, target_class]
            score.backward()
        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Falha ao capturar ativações/gradientes do Grad-CAM.")
        feature_maps = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_maps).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam.detach().float().cpu().numpy()
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        probs = torch.softmax(logits.detach().float(), dim=1)[0].cpu().numpy()
        return cam, probs
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def choose_correct_examples(df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    """Escolhe um exemplo correto por classe/plano, próximo da confiança mediana."""
    selected = []
    correct = df[df["correct"]].copy()
    for plane in PLANE_ORDER:
        for class_name in class_names:
            group = correct[
                (correct["plane_code"] == plane) & (correct["true_class"] == class_name)
            ]
            if group.empty:
                continue
            median_conf = float(group["confidence"].median())
            chosen = group.iloc[(group["confidence"] - median_conf).abs().argmin()]
            selected.append(chosen)
    return pd.DataFrame(selected).reset_index(drop=True) if selected else pd.DataFrame()


def save_correct_gradcams(model, samples, class_names, image_size, mean, std, device):
    if samples.empty:
        print("[AVISO] Não há exemplos corretos suficientes para o painel Grad-CAM.")
        return
    fig, axes = plt.subplots(len(samples), 2, figsize=(10, max(4, 3.2 * len(samples))), squeeze=False)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    for row_idx, row in samples.iterrows():
        image_path = Path(row["image_path"])
        tensor, display = prepare_image(image_path, image_size, mean, std, device)
        target_idx = class_to_idx[row["true_class"]]
        cam, probs = compute_gradcam(model, tensor, target_idx)
        plane_label = PLANE_LABELS.get(row["plane_code"], "Não identificado")
        axes[row_idx, 0].imshow(display)
        axes[row_idx, 0].set_title(f"Original — {row['true_class']}\n{plane_label}", fontsize=9)
        axes[row_idx, 1].imshow(display)
        axes[row_idx, 1].imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
        axes[row_idx, 1].set_title(
            f"Grad-CAM — classe verdadeira/prevista\nconfiança={float(probs[target_idx]):.1%}",
            fontsize=9,
        )
        for ax in axes[row_idx]:
            ax.axis("off")
    fig.suptitle("Grad-CAM de exemplos corretos por classe e plano", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    output = RESULTS_DIR / "gradcam_correct_by_plane.png"
    fig.savefig(output, dpi=145, bbox_inches="tight")
    plt.close(fig)
    print("Grad-CAM dos exemplos corretos:", output)


def save_error_gradcams(model, errors, class_names, image_size, mean, std, device):
    if errors.empty:
        print("Nenhum erro de classificação no teste; painel comparativo não foi criado.")
        return
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    fig, axes = plt.subplots(len(errors), 3, figsize=(13, max(4, 3.0 * len(errors))), squeeze=False)
    for row_idx, row in errors.reset_index(drop=True).iterrows():
        image_path = Path(row["image_path"])
        tensor, display = prepare_image(image_path, image_size, mean, std, device)
        true_idx = class_to_idx[row["true_class"]]
        pred_idx = class_to_idx[row["predicted_class"]]
        cam_true, probs = compute_gradcam(model, tensor, true_idx)
        cam_pred, _ = compute_gradcam(model, tensor, pred_idx)
        plane_label = PLANE_LABELS.get(row["plane_code"], "Não identificado")

        axes[row_idx, 0].imshow(display)
        axes[row_idx, 0].set_title(
            f"Original\nReal: {row['true_class']} | {plane_label}", fontsize=8
        )
        axes[row_idx, 1].imshow(display)
        axes[row_idx, 1].imshow(cam_true, cmap="jet", alpha=0.42, vmin=0, vmax=1)
        axes[row_idx, 1].set_title(
            f"Grad-CAM classe real\n{row['true_class']} | score={probs[true_idx]:.1%}",
            fontsize=8,
        )
        axes[row_idx, 2].imshow(display)
        axes[row_idx, 2].imshow(cam_pred, cmap="jet", alpha=0.42, vmin=0, vmax=1)
        axes[row_idx, 2].set_title(
            f"Grad-CAM classe prevista\n{row['predicted_class']} | score={probs[pred_idx]:.1%}",
            fontsize=8,
        )
        for ax in axes[row_idx]:
            ax.axis("off")
    fig.suptitle("Erros do teste: evidências para a classe real versus prevista", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    output = RESULTS_DIR / "gradcam_errors_true_vs_pred.png"
    fig.savefig(output, dpi=135, bbox_inches="tight")
    plt.close(fig)
    print("Grad-CAM comparativo dos erros:", output)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(device)
    dataset, loader, class_names, image_size, mean, std = build_test_loader(checkpoint)
    model = build_model(checkpoint, class_names, device)

    print("\n=== INVESTIGAÇÃO POR PLANO ANATÔMICO — BRISC ===")
    print("Dispositivo:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Plano extraído do sufixo do arquivo: ax=axial, co=coronal, sa=sagital")

    predictions = evaluate_test_by_plane(model, dataset, loader, class_names, device)
    unknown_count = int((predictions["plane_code"] == "unknown").sum())
    if unknown_count:
        print(f"\n[AVISO] {unknown_count} nomes de arquivo não tiveram plano identificado.")
        print(predictions.loc[predictions["plane_code"] == "unknown", "filename"].head(10).to_string(index=False))

    summary, class_metrics, class_counts = make_orientation_tables(predictions, class_names)
    predictions_path = RESULTS_DIR / "test_predictions_by_plane.csv"
    split_distribution = scan_split_plane_distribution(class_names)
    split_distribution_path = RESULTS_DIR / "plane_class_distribution.csv"
    split_distribution.to_csv(split_distribution_path, index=False)
    save_split_distribution_plot(split_distribution)
    counts_path = RESULTS_DIR / "orientation_counts.csv"
    metrics_path = RESULTS_DIR / "orientation_metrics.csv"
    class_metrics_path = RESULTS_DIR / "orientation_class_metrics.csv"
    predictions.to_csv(predictions_path, index=False)
    class_counts.to_csv(counts_path, index=False)
    summary.to_csv(metrics_path, index=False)
    class_metrics.to_csv(class_metrics_path, index=False)

    print("\n=== DISTRIBUIÇÃO POR PARTIÇÃO, PLANO E CLASSE ===")
    print(split_distribution.pivot_table(
        index=["split", "class"], columns="plane", values="n_images", fill_value=0
    ).to_string())
    print("\n=== DISTRIBUIÇÃO DO TESTE POR PLANO E CLASSE REAL ===")
    counts_pivot = predictions.pivot_table(
        index="true_class", columns="plane", values="filename", aggfunc="count", fill_value=0
    )
    print(counts_pivot.to_string())
    print("\n=== DESEMPENHO POR PLANO ===")
    if not summary.empty:
        print(summary[["plane", "n_images", "accuracy", "macro_precision", "macro_recall", "macro_f1", "errors"]].to_string(index=False, formatters={
            "accuracy": "{:.4f}".format,
            "macro_precision": "{:.4f}".format,
            "macro_recall": "{:.4f}".format,
            "macro_f1": "{:.4f}".format,
        }))
    print("\n=== RECALL POR CLASSE E PLANO ===")
    if not class_metrics.empty:
        print(class_metrics.pivot(index="class", columns="plane", values="recall").to_string(float_format=lambda x: f"{x:.4f}"))

    if unknown_count == 0:
        save_orientation_plots(predictions, summary, class_names)
    else:
        print("[AVISO] Gráficos por orientação conhecidos foram gerados apenas para ax/co/sa; verifique nomes desconhecidos.")
        save_orientation_plots(predictions, summary, class_names)

    # Seleciona exemplos corretos por combinação classe/plano e todos os erros.
    correct_samples = choose_correct_examples(predictions, class_names)
    errors = predictions[~predictions["correct"]].copy()
    save_correct_gradcams(model, correct_samples, class_names, image_size, mean, std, device)
    save_error_gradcams(model, errors, class_names, image_size, mean, std, device)

    summary_json = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "test_images": int(len(predictions)),
        "unknown_orientation_images": unknown_count,
        "orientation_metrics": summary.to_dict(orient="records"),
        "note": (
            "A análise por plano é uma estratificação descritiva por imagem. "
            "Grad-CAM é uma explicação visual aproximada e não uma máscara de tumor."
        ),
    }
    json_path = RESULTS_DIR / "orientation_investigation.json"
    json_path.write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ARQUIVOS GERADOS ===")
    for path in [
        predictions_path, split_distribution_path, counts_path, metrics_path, class_metrics_path,
        RESULTS_DIR / "plane_distribution_by_split.png",
        RESULTS_DIR / "orientation_accuracy_f1.png",
        RESULTS_DIR / "orientation_confusion_matrices.png",
        RESULTS_DIR / "gradcam_correct_by_plane.png",
        RESULTS_DIR / "gradcam_errors_true_vs_pred.png",
        json_path,
    ]:
        if path.exists():
            print(path)
    print("\nInvestigação concluída. Nenhum peso ou parâmetro foi atualizado.")


if __name__ == "__main__":
    main()
