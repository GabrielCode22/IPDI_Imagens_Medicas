#!/usr/bin/env python3
"""Carregamento e preparação do BRISC para classificação com ResNet18.

Estrutura esperada:
  data/brisc2025/classification_task/{train,test}/{glioma,meningioma,no_tumor,pituitary}

O conjunto oficial de teste permanece separado. A validação é retirada
estratificadamente dos 5.000 exemplos da pasta train.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data" / "brisc2025" / "classification_task"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGE_SIZE = 224
BATCH_SIZE = 16
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
NUM_WORKERS = 2


class ResizeWithPadding:
    """Redimensiona preservando a proporção e completa com fundo preto."""

    def __init__(self, size: int = 224) -> None:
        if size <= 0:
            raise ValueError("O tamanho da imagem deve ser positivo.")
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        left = (self.size - image.width) // 2
        top = (self.size - image.height) // 2
        canvas.paste(image, (left, top))
        return canvas


def _make_transform():
    # Sem aumento artificial nesta primeira linha de base. Primeiro medimos
    # o desempenho com uma preparação simples e reproduzível.
    return transforms.Compose(
        [
            ResizeWithPadding(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def _count_labels(labels, indices, class_names):
    counts = Counter(int(labels[i]) for i in indices)
    return {class_names[i]: counts.get(i, 0) for i in range(len(class_names))}


def build_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    train_dir = DATA_ROOT / "train"
    test_dir = DATA_ROOT / "test"

    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            "Pastas do dataset não encontradas. Esperado: "
            f"{train_dir} e {test_dir}"
        )

    # Dataset-base usado apenas para obter caminhos e rótulos para a divisão.
    base_train = datasets.ImageFolder(train_dir)
    if len(base_train) == 0:
        raise RuntimeError(f"Nenhuma imagem encontrada em {train_dir}")

    class_names = base_train.classes
    expected_classes = {"glioma", "meningioma", "no_tumor", "pituitary"}
    if set(class_names) != expected_classes:
        raise ValueError(
            f"Classes inesperadas: {class_names}. Esperadas: {sorted(expected_classes)}"
        )

    labels = base_train.targets
    indices = list(range(len(base_train)))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    transform = _make_transform()
    train_full = datasets.ImageFolder(train_dir, transform=transform)
    val_full = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    # Garante que os índices estratificados apontam para as mesmas imagens.
    if train_full.samples != val_full.samples or train_full.samples != base_train.samples:
        raise RuntimeError("A ordem dos arquivos mudou entre as leituras do dataset.")
    if test_dataset.classes != class_names:
        raise ValueError(
            f"Classes de teste ({test_dataset.classes}) diferem das de treino ({class_names})."
        )

    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    print("\n=== PREPARAÇÃO DO DATASET ===")
    print("Classes (índice -> nome):", dict(enumerate(class_names)))
    print(f"Treino:     {len(train_dataset)} imagens")
    print(f"Validação:  {len(val_dataset)} imagens")
    print(f"Teste:      {len(test_dataset)} imagens")
    print("\nDistribuição por classe:")
    print("  Treino:    ", _count_labels(labels, train_indices, class_names))
    print("  Validação: ", _count_labels(labels, val_indices, class_names))
    print("  Teste:     ", _count_labels(test_dataset.targets, range(len(test_dataset)), class_names))
    print(f"\nResolução de entrada: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Dispositivo disponível: {'CUDA/GPU' if torch.cuda.is_available() else 'CPU'}")

    return train_loader, val_loader, test_loader, class_names


def save_preview(loader: DataLoader, class_names: list[str]) -> Path:
    """Salva uma grade de imagens já transformadas, após desfazer a normalização."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    images, labels = next(iter(loader))

    mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    images = (images.cpu() * std + mean).clamp(0, 1)

    count = min(12, len(images))
    cols = 4
    rows = (count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < count:
            ax.imshow(images[i].permute(1, 2, 0).numpy())
            ax.set_title(class_names[int(labels[i])], fontsize=10)

    fig.suptitle("Amostras após pré-processamento — BRISC", fontsize=14)
    fig.tight_layout()
    output_path = RESULTS_DIR / "preprocess_preview.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    train_loader, val_loader, test_loader, class_names = build_dataloaders()

    images, labels = next(iter(train_loader))
    print("\n=== TESTE DE UM BATCH ===")
    print("Formato das imagens:", tuple(images.shape))
    print("Formato dos rótulos:", tuple(labels.shape))
    print("Tipo dos pixels:", images.dtype)
    print("Valores normalizados: min={:.3f}, max={:.3f}".format(images.min().item(), images.max().item()))

    preview = save_preview(train_loader, class_names)
    print(f"\nPré-visualização salva em: {preview}")
    print("\nCarregamento e pré-processamento concluídos.")


if __name__ == "__main__":
    main()
