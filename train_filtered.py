#!/usr/bin/env python3
"""Treina uma ResNet18 com a divisão BRISC filtrada e agrupada.

Lê os manifests produzidos por prepare_leakage_filtered_split.py e mantém
separados os artefatos do experimento original.

Execução na raiz do projeto:
    python train_filtered.py

Opções úteis:
    python train_filtered.py --epochs 20 --patience 5
    python train_filtered.py --device cpu
    python train_filtered.py --no-pretrained  # somente sem acesso aos pesos ImageNet

Este script NÃO lê nem utiliza o conjunto oficial de teste.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm

from data_pipeline import ResizeWithPadding, IMAGE_SIZE


ROOT = Path(__file__).resolve().parent
FILTERED_DIR = ROOT / "results" / "leakage_filtered"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = FILTERED_DIR
TRAIN_MANIFEST = FILTERED_DIR / "train_manifest.csv"
VAL_MANIFEST = FILTERED_DIR / "validation_manifest.csv"
EXPECTED_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ManifestImageDataset(Dataset):
    """Dataset que carrega caminhos e rótulos diretamente de um manifesto."""

    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        path = Path(str(row["path"]))
        label = int(row["class_idx"])
        try:
            with Image.open(path) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
        except Exception as exc:
            raise RuntimeError(f"Não foi possível abrir a imagem: {path}") from exc
        return self.transform(image), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina a ResNet18 com os manifests BRISC filtrados e agrupados."
    )
    parser.add_argument("--epochs", type=int, default=15,
                        help="Número máximo de épocas (padrão: 15).")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Taxa de aprendizado (padrão: 0.0001).")
    parser.add_argument("--patience", type=int, default=5,
                        help="Épocas sem melhora na validação (padrão: 5).")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Tamanho do lote (padrão: 16).")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Processos para carregamento de dados (padrão: 2).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semente aleatória (padrão: 42).")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto",
                        help="Dispositivo: auto, cuda ou cpu (padrão: auto).")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Não utilizar pesos pré-treinados do ImageNet.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda foi solicitado, mas CUDA não está disponível.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_manifests() -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = [str(p) for p in (TRAIN_MANIFEST, VAL_MANIFEST) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Manifestos filtrados não encontrados:\n" + "\n".join(missing)
            + "\nExecute primeiro: python prepare_leakage_filtered_split.py"
        )

    train_df = pd.read_csv(TRAIN_MANIFEST)
    val_df = pd.read_csv(VAL_MANIFEST)
    required = {"path", "class", "class_idx", "component_id", "split"}
    for name, frame in (("treino", train_df), ("validação", val_df)):
        absent = required - set(frame.columns)
        if absent:
            raise ValueError(f"Manifesto de {name} sem colunas: {sorted(absent)}")
        if frame.empty:
            raise ValueError(f"Manifesto de {name} está vazio.")
        frame["path"] = frame["path"].astype(str).map(lambda p: str(Path(p).expanduser().resolve()))
        frame["class"] = frame["class"].astype(str)
        frame["class_idx"] = pd.to_numeric(frame["class_idx"], errors="raise").astype(int)
        frame["component_id"] = frame["component_id"].astype(str)
        frame["split"] = frame["split"].astype(str)

    class_to_idx = {name: idx for idx, name in enumerate(EXPECTED_CLASSES)}
    for name, frame, split_value in (
        ("treino", train_df, "fit_train"),
        ("validação", val_df, "validation"),
    ):
        if set(frame["class"]) != set(EXPECTED_CLASSES):
            raise ValueError(f"Classes inesperadas no manifesto de {name}: {sorted(set(frame['class']))}")
        expected_idx = frame["class"].map(class_to_idx)
        if expected_idx.isna().any() or not np.array_equal(expected_idx.to_numpy(), frame["class_idx"].to_numpy()):
            raise ValueError(f"Índices de classe incompatíveis no manifesto de {name}.")
        if set(frame["split"]) != {split_value}:
            raise ValueError(f"Coluna split incorreta no manifesto de {name}.")
        if frame["path"].duplicated().any():
            raise ValueError(f"Há caminhos duplicados no manifesto de {name}.")
        missing_files = [p for p in frame["path"] if not Path(p).is_file()]
        if missing_files:
            raise FileNotFoundError(
                f"{len(missing_files)} imagens do manifesto de {name} não existem. "
                f"Exemplo: {missing_files[:3]}"
            )
        wrong_parent = [p for p, c in zip(frame["path"], frame["class"]) if Path(p).parent.name != c]
        if wrong_parent:
            raise ValueError(f"Pasta/classe inconsistente no manifesto de {name}: {wrong_parent[:3]}")

    train_paths = set(train_df["path"])
    val_paths = set(val_df["path"])
    if train_paths & val_paths:
        raise RuntimeError("Há imagens repetidas simultaneamente em treino e validação.")
    train_components = set(train_df["component_id"])
    val_components = set(val_df["component_id"])
    overlap = train_components & val_components
    if overlap:
        raise RuntimeError(f"Há {len(overlap)} componentes compartilhados entre treino e validação.")

    return train_df, val_df


def make_transform() -> transforms.Compose:
    return transforms.Compose([
        ResizeWithPadding(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    transform = make_transform()
    train_dataset = ManifestImageDataset(train_df, transform)
    val_dataset = ManifestImageDataset(val_df, transform)
    generator = torch.Generator().manual_seed(42)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_loader, val_loader


def build_model(num_classes: int, use_pretrained: bool) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if use_pretrained else None
    try:
        model = resnet18(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Não foi possível carregar os pesos pré-treinados da ResNet18. "
            "Verifique a conexão com a internet ou execute com --no-pretrained."
        ) from exc
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    epoch: int,
    total_epochs: int,
    phase: str,
    use_amp: bool,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    all_targets: list[int] = []
    all_predictions: list[int] = []

    progress = tqdm(loader, desc=f"Época {epoch:02d}/{total_epochs:02d} — {phase}", leave=False)
    for images, targets in progress:
        images = images.to(device, non_blocking=(device.type == "cuda"))
        targets = targets.to(device, non_blocking=(device.type == "cuda"))
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        batch_size = targets.size(0)
        total_loss += float(loss.detach().item()) * batch_size
        all_targets.extend(targets.detach().cpu().tolist())
        all_predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        progress.set_postfix(loss=f"{loss.item():.4f}")

    mean_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_targets, all_predictions)
    macro_f1 = f1_score(all_targets, all_predictions, average="macro", zero_division=0)
    return mean_loss, float(accuracy), float(macro_f1)


def save_curves(history: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="Treino")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validação")
    axes[0].set_title("Perda — treinamento filtrado")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(history["epoch"], history["train_accuracy"], label="Treino")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validação")
    axes[1].set_title("Acurácia — treinamento filtrado")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Acurácia")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves_filtered.png", dpi=160)
    plt.close(fig)


def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    use_amp: bool,
) -> None:
    model.eval()
    targets_all: list[int] = []
    predictions_all: list[int] = []
    probabilities_all: list[list[float]] = []
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc="Avaliando validação filtrada", leave=False):
            images = images.to(device, non_blocking=(device.type == "cuda"))
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
            probs = torch.softmax(logits.float(), dim=1)
            predictions = probs.argmax(dim=1)
            targets_all.extend(targets.tolist())
            predictions_all.extend(predictions.cpu().tolist())
            probabilities_all.extend(probs.cpu().tolist())

    report = classification_report(
        targets_all,
        predictions_all,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    (RESULTS_DIR / "validation_classification_report_filtered.txt").write_text(report, encoding="utf-8")

    pred_df = pd.DataFrame({
        "true_class": [class_names[i] for i in targets_all],
        "predicted_class": [class_names[i] for i in predictions_all],
        "confidence": [max(row) for row in probabilities_all],
    })
    for idx, class_name in enumerate(class_names):
        pred_df[f"prob_{class_name}"] = [row[idx] for row in probabilities_all]
    pred_df.to_csv(RESULTS_DIR / "validation_predictions_filtered.csv", index=False)

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
    ax.set_title("Matriz de confusão — validação filtrada")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "validation_confusion_matrix_filtered.png", dpi=160)
    plt.close(fig)

    print("\n=== RELATÓRIO DE CLASSIFICAÇÃO — VALIDAÇÃO FILTRADA ===")
    print(report)


def main() -> int:
    args = parse_args()
    if args.epochs < 1 or args.lr <= 0 or args.patience < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("epochs, lr, patience e batch-size devem ser positivos; num-workers não pode ser negativo.")

    set_seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    use_amp = device.type == "cuda"
    print("\n=== TREINAMENTO FILTRADO — RESNET18 / BRISC ===")
    print("Dispositivo:", device)
    if use_amp:
        print("GPU:", torch.cuda.get_device_name(0))
        print("Precisão mista (AMP): ativada")
    else:
        print("AMP: desativada (execução CPU)")
    print("Pesos pré-treinados:", not args.no_pretrained)

    train_df, val_df = load_and_validate_manifests()
    class_names = EXPECTED_CLASSES.copy()
    train_counts = train_df["class"].value_counts().reindex(class_names, fill_value=0).to_dict()
    val_counts = val_df["class"].value_counts().reindex(class_names, fill_value=0).to_dict()
    print("\n=== MANIFESTOS VALIDADOS ===")
    print(f"Treino: {len(train_df)} imagens | {train_counts}")
    print(f"Validação: {len(val_df)} imagens | {val_counts}")
    print("Componentes treino/validação em comum: 0")
    print("Teste oficial: não carregado nem utilizado nesta etapa")

    train_loader, val_loader = build_loaders(
        train_df, val_df, args.batch_size, args.num_workers, device
    )
    model = build_model(len(class_names), use_pretrained=not args.no_pretrained).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    rows: list[dict[str, float | int]] = []
    checkpoint_path = CHECKPOINT_DIR / "best_resnet18_filtered.pth"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler,
            epoch, args.epochs, "treino", use_amp,
        )
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, device, None, scaler,
            epoch, args.epochs, "validação", use_amp,
        )
        scheduler.step(val_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
            "learning_rate": current_lr,
        })
        print(
            f"Época {epoch:02d}/{args.epochs:02d} | "
            f"train loss={train_loss:.4f}, acc={train_acc:.4f}, F1-macro={train_f1:.4f} | "
            f"val loss={val_loss:.4f}, acc={val_acc:.4f}, F1-macro={val_f1:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
            checkpoint = {
                "architecture": "resnet18",
                "model_state_dict": best_state,
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "normalization_mean": list(MEAN),
                "normalization_std": list(STD),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "seed": args.seed,
                "training_protocol": "BRISC filtered pool; exact SHA-256 and connected pHash candidates touching official test excluded; StratifiedGroupKFold train/validation",
                "train_manifest_sha256": sha256_file(TRAIN_MANIFEST),
                "validation_manifest_sha256": sha256_file(VAL_MANIFEST),
                "excluded_train_candidates": 1191,
                "phash_hamming_threshold": 6,
                "warning": "pHash is visual-similarity triage, not patient identity; official test can contain internal related images.",
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  Melhor modelo filtrado salvo em {checkpoint_path.relative_to(ROOT)}")
        else:
            epochs_without_improvement += 1

        history_df = pd.DataFrame(rows)
        history_df.to_csv(RESULTS_DIR / "training_history_filtered.csv", index=False)
        save_curves(history_df)

        if epochs_without_improvement >= args.patience:
            print(f"\nParada antecipada: validação não melhorou por {args.patience} épocas.")
            break

    if best_state is None:
        raise RuntimeError("Treinamento terminou sem produzir checkpoint filtrado.")

    model.load_state_dict(best_state)
    print(f"\nMelhor época: {best_epoch} | menor val loss: {best_val_loss:.4f}")
    evaluate_validation(model, val_loader, class_names, device, use_amp)

    summary = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if use_amp else None,
        "train_images": len(train_df),
        "validation_images": len(val_df),
        "train_class_counts": train_counts,
        "validation_class_counts": val_counts,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(rows),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "patience": args.patience,
        "amp_enabled": use_amp,
        "checkpoint": str(checkpoint_path),
        "train_manifest_sha256": sha256_file(TRAIN_MANIFEST),
        "validation_manifest_sha256": sha256_file(VAL_MANIFEST),
        "official_test_used": False,
    }
    (RESULTS_DIR / "training_summary_filtered.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n=== ARQUIVOS GERADOS ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Histórico:  {RESULTS_DIR / 'training_history_filtered.csv'}")
    print(f"Curvas:     {RESULTS_DIR / 'training_curves_filtered.png'}")
    print(f"Matriz:     {RESULTS_DIR / 'validation_confusion_matrix_filtered.png'}")
    print(f"Relatório:  {RESULTS_DIR / 'validation_classification_report_filtered.txt'}")
    print("\nTreinamento filtrado concluído. O teste oficial NÃO foi utilizado.")
    print("O checkpoint original best_resnet18.pth não foi alterado.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        raise SystemExit(1)
