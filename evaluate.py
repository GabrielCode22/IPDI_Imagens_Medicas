#!/usr/bin/env python3
"""Avalia uma única vez o melhor checkpoint no teste oficial do BRISC.

Execute na raiz do projeto, com o ambiente virtual ativado:
    python evaluate.py

O script não treina, não ajusta hiperparâmetros e não altera o checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

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
from torchvision.models import resnet18
from tqdm import tqdm

from data_pipeline import build_dataloaders


ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_resnet18.pth"
RESULTS_DIR = ROOT / "results"


def load_checkpoint(path: Path, device: torch.device) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {path}\nExecute primeiro: python train.py"
        )
    # weights_only=True é mais seguro; fallback mantém compatibilidade com
    # versões antigas do PyTorch que ainda não aceitam esse argumento.
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint inválido: model_state_dict não encontrado.")
    return checkpoint


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== AVALIAÇÃO FINAL — TESTE OFICIAL BRISC ===")
    print("Dispositivo:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Recupera o mesmo pipeline e as mesmas transformações usados no treino.
    _train_loader, _val_loader, test_loader, pipeline_classes = build_dataloaders()

    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    class_names = checkpoint.get("class_names", pipeline_classes)
    if class_names != pipeline_classes:
        raise ValueError(
            "A ordem das classes no checkpoint difere da ordem do dataset.\n"
            f"Checkpoint: {class_names}\nPipeline: {pipeline_classes}"
        )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []

    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for images, targets in tqdm(test_loader, desc="Teste oficial", leave=True):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(images)
            probabilities = torch.softmax(logits.float(), dim=1)
            predictions = probabilities.argmax(dim=1)

            all_targets.extend(targets.tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    labels = list(range(len(class_names)))
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_precision = precision_score(
        all_targets, all_predictions, labels=labels, average="macro", zero_division=0
    )
    macro_recall = recall_score(
        all_targets, all_predictions, labels=labels, average="macro", zero_division=0
    )
    macro_f1 = f1_score(
        all_targets, all_predictions, labels=labels, average="macro", zero_division=0
    )
    report = classification_report(
        all_targets,
        all_predictions,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    matrix = confusion_matrix(all_targets, all_predictions, labels=labels)

    # ImageFolder mantém a ordem dos caminhos; o DataLoader de teste não embaralha.
    test_dataset = test_loader.dataset
    samples = getattr(test_dataset, "samples", None)
    if samples is None or len(samples) != len(all_targets):
        paths = [""] * len(all_targets)
    else:
        paths = [sample[0] for sample in samples]

    probabilities_array = np.asarray(all_probabilities)
    rows = []
    for index, (true_id, pred_id, probs, path) in enumerate(
        zip(all_targets, all_predictions, probabilities_array, paths)
    ):
        row = {
            "index": index,
            "image_path": path,
            "true_class": class_names[true_id],
            "predicted_class": class_names[pred_id],
            "correct": bool(true_id == pred_id),
            "confidence": float(probs[pred_id]),
        }
        for class_index, class_name in enumerate(class_names):
            row[f"prob_{class_name}"] = float(probs[class_index])
        rows.append(row)

    predictions_path = RESULTS_DIR / "test_predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_path, index=False)

    report_path = RESULTS_DIR / "test_classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    matrix_path = RESULTS_DIR / "test_confusion_matrix.png"
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    ).plot(
        ax=ax,
        cmap="Blues",
        values_format="d",
        xticks_rotation=30,
        colorbar=False,
    )
    ax.set_title("Matriz de confusão — teste oficial BRISC")
    fig.tight_layout()
    fig.savefig(matrix_path, dpi=160)
    plt.close(fig)

    metrics = {
        "evaluation": "official_test",
        "num_images": len(all_targets),
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "checkpoint_best_epoch": checkpoint.get("best_epoch"),
        "checkpoint_best_validation_loss": checkpoint.get("best_val_loss"),
        "class_names": class_names,
    }
    metrics_path = RESULTS_DIR / "test_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nCheckpoint utilizado:", CHECKPOINT_PATH)
    print("Melhor época de validação:", checkpoint.get("best_epoch", "não informado"))
    print("\n=== MÉTRICAS GERAIS — TESTE OFICIAL ===")
    print(f"Imagens:             {len(all_targets)}")
    print(f"Acurácia:            {accuracy:.4f}")
    print(f"Precisão macro:      {macro_precision:.4f}")
    print(f"Recall macro:        {macro_recall:.4f}")
    print(f"F1-score macro:      {macro_f1:.4f}")
    print("\n=== RELATÓRIO POR CLASSE ===")
    print(report)
    print("Matriz de confusão (linhas = classe real; colunas = previsão):")
    print(matrix)

    print("\n=== ARQUIVOS GERADOS ===")
    print("Métricas:       ", metrics_path)
    print("Relatório:      ", report_path)
    print("Matriz:         ", matrix_path)
    print("Previsões CSV:  ", predictions_path)
    print("\nAvaliação concluída. Nenhum peso ou parâmetro foi atualizado.")


if __name__ == "__main__":
    main()
