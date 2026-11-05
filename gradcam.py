#!/usr/bin/env python3
"""Gera visualizações Grad-CAM para exemplos do teste oficial BRISC.

Requisitos:
- Treinamento concluído: checkpoints/best_resnet18.pth
- Avaliação concluída: results/test_predictions.csv

Execução na raiz do projeto:
    python gradcam.py
    python gradcam.py --max-errors 8

O script não treina nem modifica os pesos. Para cada exemplo, o mapa explica
as regiões que contribuíram para a classe prevista pelo modelo.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18

from data_pipeline import DATA_ROOT, IMAGE_SIZE, ResizeWithPadding

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_resnet18.pth"
PREDICTIONS_PATH = ROOT / "results" / "test_predictions.csv"
OUTPUT_PATH = ROOT / "results" / "gradcam_examples.png"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualiza explicações Grad-CAM do BRISC.")
    parser.add_argument(
        "--max-errors", type=int, default=4,
        help="Número máximo de erros do teste a incluir (padrão: 4).",
    )
    parser.add_argument(
        "--correct-per-class", type=int, default=1,
        help="Exemplos corretos por classe (padrão: 1).",
    )
    return parser.parse_args()


def load_checkpoint(device: torch.device) -> tuple[nn.Module, list[str]]:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {CHECKPOINT_PATH}. Execute python train.py primeiro."
        )
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

    class_names = checkpoint.get(
        "class_names", ["glioma", "meningioma", "no_tumor", "pituitary"]
    )
    if not isinstance(class_names, list) or len(class_names) != 4:
        raise ValueError("Lista de classes inválida no checkpoint.")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


def select_examples(
    predictions: pd.DataFrame,
    class_names: list[str],
    correct_per_class: int,
    max_errors: int,
) -> pd.DataFrame:
    required = {"image_path", "true_class", "predicted_class", "correct", "confidence"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {PREDICTIONS_PATH}: {sorted(missing)}")

    selected_indices: list[int] = []
    # Seleciona exemplos corretos de maior confiança, um conjunto por classe.
    for class_name in class_names:
        correct = predictions[
            (predictions["true_class"] == class_name)
            & predictions["correct"].astype(bool)
        ].sort_values("confidence", ascending=False)
        selected_indices.extend(correct.head(correct_per_class).index.tolist())

    # Inclui erros de menor confiança, úteis para examinar previsões difíceis.
    errors = predictions[~predictions["correct"].astype(bool)].sort_values(
        "confidence", ascending=True
    )
    selected_indices.extend(errors.head(max_errors).index.tolist())

    selected = predictions.loc[selected_indices].copy()
    selected = selected.drop_duplicates(subset="image_path")
    if selected.empty:
        raise RuntimeError("Não foram encontrados exemplos para visualizar.")
    return selected.reset_index(drop=True)


def make_input_and_display(path: Path, device: torch.device) -> tuple[torch.Tensor, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Imagem não encontrada: {path}")
    image = Image.open(path).convert("RGB")
    resized = ResizeWithPadding(IMAGE_SIZE)(image)
    display = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = transforms.ToTensor()(resized)
    tensor = transforms.Normalize(MEAN, STD)(tensor).unsqueeze(0).to(device)
    return tensor, display


def compute_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def save_activation(_module, _inputs, output: torch.Tensor) -> None:
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    target_layer = model.layer4[-1]
    handle = target_layer.register_forward_hook(save_activation)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            score = logits[0, target_class]
            score.backward()

        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Não foi possível obter ativações/gradientes para Grad-CAM.")

        feature_maps = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_maps).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam.detach().cpu().numpy()
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        probabilities = torch.softmax(logits.detach().float(), dim=1)[0].cpu().numpy()
        return cam, probabilities
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def main() -> None:
    args = parse_args()
    if args.max_errors < 0 or args.correct_per_class < 0:
        raise ValueError("Os limites de exemplos não podem ser negativos.")
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {PREDICTIONS_PATH}. Execute python evaluate.py primeiro."
        )

    RESULTS_DIR = ROOT / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictions = pd.read_csv(PREDICTIONS_PATH)
    model, class_names = load_checkpoint(device)
    selected = select_examples(
        predictions, class_names, args.correct_per_class, args.max_errors
    )

    print("\n=== GRAD-CAM — EXEMPLOS DO TESTE OFICIAL ===")
    print("Dispositivo:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"Exemplos selecionados: {len(selected)}")

    rows = len(selected)
    fig, axes = plt.subplots(rows, 2, figsize=(9, max(4, 3.6 * rows)), squeeze=False)

    for row_index, row in selected.iterrows():
        image_path = Path(str(row["image_path"]))
        input_tensor, display_image = make_input_and_display(image_path, device)

        # Explica a classe prevista (inclusive nos casos classificados incorretamente).
        predicted_class = class_names.index(str(row["predicted_class"]))
        cam, probabilities = compute_gradcam(model, input_tensor, predicted_class)
        confidence = float(probabilities[predicted_class])
        true_name = str(row["true_class"])
        predicted_name = class_names[predicted_class]
        status = "CORRETO" if true_name == predicted_name else "ERRO"

        axes[row_index, 0].imshow(display_image)
        axes[row_index, 0].set_title(f"Original\nReal: {true_name}", fontsize=10)
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(display_image)
        axes[row_index, 1].imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
        axes[row_index, 1].set_title(
            f"Grad-CAM — {predicted_name}\n{status} | confiança: {confidence:.1%}",
            fontsize=10,
        )
        axes[row_index, 1].axis("off")

        print(
            f"{row_index + 1:02d}. real={true_name:<10} "
            f"prevista={predicted_name:<10} confiança={confidence:.3f} [{status}]"
        )

    fig.suptitle(
        "Grad-CAM — regiões que influenciaram a classe prevista (BRISC)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("\nImagem salva em:", OUTPUT_PATH)
    print("Observação: Grad-CAM destaca evidências usadas pela rede; não é uma máscara clínica do tumor.")


if __name__ == "__main__":
    main()
