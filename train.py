#!/usr/bin/env python3
"""Treina uma ResNet18 para classificar as quatro categorias do BRISC.

Usa a divisão treino/validação criada por data_pipeline.py. O conjunto oficial
 de teste é carregado pelo pipeline, mas NÃO é usado durante o treinamento.

Execução, na raiz do projeto:
    python train.py

Exemplos:
    python train.py --epochs 20 --lr 0.0001
    python train.py --no-pretrained   # se não houver internet para baixar pesos
"""
from __future__ import annotations

import argparse
import copy
import random
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
    f1_score,
)
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from data_pipeline import build_dataloaders


ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treinamento da ResNet18 para classificação do BRISC."
    )
    parser.add_argument("--epochs", type=int, default=15,
                        help="Número máximo de épocas (padrão: 15).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Taxa de aprendizado (padrão: 0.0001).")
    parser.add_argument("--patience", type=int, default=5,
                        help="Épocas sem melhora na validação antes de parar (padrão: 5).")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Não baixar pesos pré-treinados do ImageNet.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente aleatória (padrão: 42).")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int, use_pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível carregar os pesos pré-treinados da ResNet18. "
            "Verifique a conexão com a internet e tente novamente. Se quiser "
            "treinar sem pesos pré-treinados, execute com --no-pretrained."
        ) from exc

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    epoch: int,
    total_epochs: int,
    phase: str,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(
        loader,
        desc=f"Época {epoch:02d}/{total_epochs:02d} — {phase}",
        leave=False,
    )

    for images, targets in progress:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
                loss = criterion(logits, targets)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        predictions = logits.detach().argmax(dim=1)
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

        progress.set_postfix(loss=f"{loss.item():.4f}")

    mean_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(
        all_targets, all_predictions, average="macro", zero_division=0
    )
    return mean_loss, float(accuracy), float(macro_f1)


def save_training_curves(history: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["epoch"], history["train_loss"], label="Treino")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validação")
    axes[0].set_title("Perda por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["train_accuracy"], label="Treino")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validação")
    axes[1].set_title("Acurácia por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=160)
    plt.close(fig)


def save_validation_results(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    class_names: list[str],
    device: torch.device,
) -> None:
    model.eval()
    targets_all: list[int] = []
    predictions_all: list[int] = []

    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Avaliando validação", leave=False):
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)
            predictions = logits.argmax(dim=1)
            targets_all.extend(targets.tolist())
            predictions_all.extend(predictions.cpu().tolist())

    report = classification_report(
        targets_all,
        predictions_all,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (RESULTS_DIR / "validation_classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        targets_all,
        predictions_all,
        labels=list(range(len(class_names))),
        display_labels=class_names,
        cmap="Blues",
        xticks_rotation=30,
        values_format="d",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Matriz de confusão — validação")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "validation_confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO — VALIDAÇÃO ===")
    print(report)


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.patience < 1:
        raise ValueError("epochs, lr e patience precisam ser positivos.")

    set_seed(args.seed)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print("\n=== TREINAMENTO DA RESNET18 ===")
    print("Dispositivo:", device)
    if use_amp:
        print("GPU:", torch.cuda.get_device_name(0))
        print("Precisão mista (AMP): ativada")
    print("Pesos pré-treinados:", not args.no_pretrained)

    train_loader, val_loader, _test_loader, class_names = build_dataloaders()
    model = build_model(len(class_names), use_pretrained=not args.no_pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler,
            epoch, args.epochs, "treino",
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device, None, scaler,
            epoch, args.epochs, "validação",
        )
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "train_macro_f1": train_f1,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_f1,
                "learning_rate": current_lr,
            }
        )

        print(
            f"Época {epoch:02d}/{args.epochs:02d} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.4f}, "
            f"F1-macro={train_f1:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}, "
            f"F1-macro={val_f1:.4f} | lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": "resnet18",
                    "model_state_dict": best_state,
                    "class_names": class_names,
                    "image_size": 224,
                    "normalization_mean": [0.485, 0.456, 0.406],
                    "normalization_std": [0.229, 0.224, 0.225],
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                    "seed": args.seed,
                },
                CHECKPOINT_DIR / "best_resnet18.pth",
            )
            print("  Melhor modelo salvo em checkpoints/best_resnet18.pth")
        else:
            epochs_without_improvement += 1

        pd.DataFrame(rows).to_csv(
            RESULTS_DIR / "training_history.csv", index=False
        )
        save_training_curves(pd.DataFrame(rows))

        if epochs_without_improvement >= args.patience:
            print(
                f"\nParada antecipada: validação não melhorou por "
                f"{args.patience} épocas."
            )
            break

    if best_state is None:
        raise RuntimeError("O treinamento terminou sem produzir um checkpoint.")

    model.load_state_dict(best_state)
    print(f"\nMelhor época: {best_epoch} | menor val loss: {best_val_loss:.4f}")
    save_validation_results(model, val_loader, class_names, device)

    print("\n=== ARQUIVOS GERADOS ===")
    print("Checkpoint: checkpoints/best_resnet18.pth")
    print("Histórico:  results/training_history.csv")
    print("Curvas:     results/training_curves.png")
    print("Matriz:     results/validation_confusion_matrix.png")
    print("Relatório:  results/validation_classification_report.txt")
    print("\nO conjunto oficial de teste NÃO foi usado nesta etapa.")


if __name__ == "__main__":
    main()
