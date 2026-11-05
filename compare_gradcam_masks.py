#!/usr/bin/env python3
"""Compara Grad-CAM da ResNet18 com máscaras tumorais do BRISC.

Executar na raiz de tumor_cnn:
    python compare_gradcam_masks.py

Estrutura esperada:
  data/brisc2025/classification_task/test/<classe>/*.jpg
  data/brisc2025/segmentation_task/test/images/*.jpg
  data/brisc2025/segmentation_task/test/masks/*.png

O script NÃO treina nem altera o checkpoint. A comparação é feita apenas
nas imagens tumorais do teste oficial que tenham máscara pareada. Imagens
no_tumor não possuem máscara correspondente no subconjunto de segmentação e
são contabilizadas separadamente.

As métricas medem alinhamento espacial aproximado entre saliência Grad-CAM e
máscara. Não são métricas de segmentação clínica, pois Grad-CAM não é um
segmentador e possui resolução espacial limitada.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
from tqdm import tqdm

from data_pipeline import DATA_ROOT, IMAGE_SIZE, ResizeWithPadding

ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = ROOT / "checkpoints" / "best_resnet18.pth"
RESULTS_DIR = ROOT / "results"
DEFAULT_SEG_ROOT = DATA_ROOT.parent / "segmentation_task"
BATCH_SIZE = 16
NUM_WORKERS = 2
MEAN_DEFAULT = (0.485, 0.456, 0.406)
STD_DEFAULT = (0.229, 0.224, 0.225)
PLANE_LABELS = {"ax": "Axial", "co": "Coronal", "sa": "Sagital"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara mapas Grad-CAM com as máscaras do BRISC no teste oficial."
    )
    parser.add_argument(
        "--seg-root", type=Path, default=DEFAULT_SEG_ROOT,
        help=f"Pasta segmentation_task (padrão: {DEFAULT_SEG_ROOT})",
    )
    parser.add_argument(
        "--max-errors", type=int, default=30,
        help="Máximo de erros incluídos no painel visual (padrão: 30).",
    )
    parser.add_argument(
        "--cam-threshold", type=float, default=0.50,
        help="Limiar da CAM normalizada para Dice/IoU (padrão: 0.50).",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {path}\nExecute python train.py primeiro."
        )
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict) or "model_state_dict" not in obj:
        raise ValueError("Checkpoint inválido: model_state_dict não encontrado.")
    return obj


def load_model(checkpoint: dict[str, Any], class_names: list[str], device: torch.device):
    saved_names = checkpoint.get("class_names", class_names)
    if list(saved_names) != list(class_names):
        raise ValueError(
            f"Ordem de classes divergente: checkpoint={saved_names}, dataset={class_names}"
        )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model


def parse_plane(filename: str) -> str:
    match = re.search(r"_(ax|co|sa)(?:_t1)?$", Path(filename).stem.lower())
    return match.group(1) if match else "unknown"


def build_image_transform(image_size: int, mean, std):
    return transforms.Compose([
        ResizeWithPadding(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def resize_mask_with_padding(mask: Image.Image, size: int) -> np.ndarray:
    """Aplica o mesmo enquadramento da imagem, usando vizinho mais próximo."""
    mask = ImageOps.exif_transpose(mask).convert("L")
    mask.thumbnail((size, size), Image.Resampling.NEAREST)
    canvas = Image.new("L", (size, size), 0)
    left = (size - mask.width) // 2
    top = (size - mask.height) // 2
    canvas.paste(mask, (left, top))
    array = np.asarray(canvas, dtype=np.uint8)
    return array > 0


def load_rgb_display(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as source:
        image = ResizeWithPadding(image_size)(source)
    return np.asarray(image, dtype=np.float32) / 255.0


def load_mask(path: Path, image_size: int) -> tuple[np.ndarray, list[int]]:
    with Image.open(path) as source:
        raw = np.asarray(ImageOps.exif_transpose(source).convert("L"), dtype=np.uint8)
        values = np.unique(raw)
        mask = resize_mask_with_padding(source, image_size)
    return mask, [int(v) for v in values[:20]]


def compute_gradcam(model, input_tensor: torch.Tensor, target_class: int):
    """Grad-CAM da camada layer4[-1], normalizado por imagem para [0, 1]."""
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    handle = model.layer4[-1].register_forward_hook(forward_hook)
    try:
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = model(input_tensor)
            score = logits[0, target_class]
            score.backward()
        if "value" not in activations or "value" not in gradients:
            raise RuntimeError("Não foi possível capturar gradientes do Grad-CAM.")
        feature_maps = activations["value"]
        grads = gradients["value"]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_maps).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam.detach().float().cpu().numpy()
        lo, hi = float(cam.min()), float(cam.max())
        cam = (cam - lo) / (hi - lo) if hi > lo else np.zeros_like(cam)
        probabilities = torch.softmax(logits.detach().float(), dim=1)[0].cpu().numpy()
        return cam.astype(np.float32), probabilities
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


def binary_metrics(cam: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float]:
    pred = cam >= threshold
    truth = mask.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "dice": (2 * tp / denom_dice) if denom_dice else 1.0,
        "iou": (tp / denom_iou) if denom_iou else 1.0,
        "precision": precision,
        "recall": recall,
        "cam_area_ratio": float(pred.mean()),
    }


def ranking_metrics(cam: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    y = mask.astype(np.uint8).ravel()
    scores = cam.astype(np.float32).ravel()
    if y.min() == y.max():
        return {"pixel_roc_auc": float("nan"), "pixel_average_precision": float("nan")}
    return {
        "pixel_roc_auc": float(roc_auc_score(y, scores)),
        "pixel_average_precision": float(average_precision_score(y, scores)),
    }


def metrics_at_top_fraction(cam: np.ndarray, mask: np.ndarray, fraction: float) -> dict[str, float]:
    cutoff = float(np.quantile(cam, 1.0 - fraction))
    values = binary_metrics(cam, mask, cutoff)
    return {
        "dice": values["dice"], "iou": values["iou"],
        "precision": values["precision"], "recall": values["recall"],
        "cam_area_ratio": values["cam_area_ratio"],
    }


def pair_test_images(
    classification_dir: Path, seg_root: Path, class_names: list[str]
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, int]]:
    """Pareia pelo basename oficial; no_tumor fica fora pois não tem máscara."""
    test_images_dir = seg_root / "test" / "images"
    test_masks_dir = seg_root / "test" / "masks"
    if not test_images_dir.is_dir() or not test_masks_dir.is_dir():
        raise FileNotFoundError(
            "Não encontrei as pastas de segmentação do teste. Esperado:\n"
            f"  {test_images_dir}\n  {test_masks_dir}\n\n"
            "Extraia a pasta segmentation_task completa do BRISC para data/brisc2025/."
        )

    seg_images = {
        p.stem: p for p in test_images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }
    seg_masks = {
        p.stem: p for p in test_masks_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    }

    rows: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    counts = {"classification_test_total": 0, "no_tumor_excluded": 0,
              "tumor_images_seen": 0, "matched_pairs": 0,
              "missing_seg_image": 0, "missing_mask": 0, "dimension_mismatch": 0}

    for class_name in class_names:
        class_dir = classification_dir / class_name
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            counts["classification_test_total"] += 1
            if class_name == "no_tumor":
                counts["no_tumor_excluded"] += 1
                rows.append({
                    "filename": image_path.name,
                    "class": class_name,
                    "segmentation_image": "",
                    "mask": "",
                    "status": "no_tumor_excluded_no_tumor_mask_in_segmentation_subset",
                })
                continue
            counts["tumor_images_seen"] += 1
            stem = image_path.stem
            seg_image_path = seg_images.get(stem)
            mask_path = seg_masks.get(stem)
            status = "matched"
            if seg_image_path is None:
                status = "missing_segmentation_image"
                counts["missing_seg_image"] += 1
            elif mask_path is None:
                status = "missing_mask"
                counts["missing_mask"] += 1
            else:
                try:
                    with Image.open(image_path) as im_a, Image.open(seg_image_path) as im_b, Image.open(mask_path) as m:
                        size_a = ImageOps.exif_transpose(im_a).size
                        size_b = ImageOps.exif_transpose(im_b).size
                        size_m = ImageOps.exif_transpose(m).size
                    if size_a != size_b or size_a != size_m:
                        status = f"dimension_mismatch:class={size_a},seg={size_b},mask={size_m}"
                        counts["dimension_mismatch"] += 1
                    else:
                        paired.append({
                            "image_path": image_path,
                            "seg_image_path": seg_image_path,
                            "mask_path": mask_path,
                            "true_class": class_name,
                            "filename": image_path.name,
                            "plane_code": parse_plane(image_path.name),
                        })
                        counts["matched_pairs"] += 1
                except Exception as exc:
                    status = f"unreadable_pair: {exc}"
                    counts["dimension_mismatch"] += 1
            rows.append({
                "filename": image_path.name,
                "class": class_name,
                "segmentation_image": str(seg_image_path) if seg_image_path else "",
                "mask": str(mask_path) if mask_path else "",
                "status": status,
            })
    return paired, pd.DataFrame(rows), counts


def analyze_pairs(
    pairs: list[dict[str, Any]], model, class_names: list[str], image_size: int,
    mean, std, device: torch.device, cam_threshold: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    image_transform = build_image_transform(image_size, mean, std)
    rows: list[dict[str, Any]] = []
    visual_data: list[dict[str, Any]] = []

    for item in tqdm(pairs, desc="Grad-CAM versus máscaras", leave=True):
        with Image.open(item["image_path"]) as src:
            display_image = ResizeWithPadding(image_size)(src)
            image_tensor = image_transform(src).unsqueeze(0).to(device)
        mask, raw_mask_values = load_mask(item["mask_path"], image_size)
        if int(mask.sum()) == 0:
            print(f"[AVISO] Máscara vazia; ignorando: {item['filename']}")
            continue

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))
        true_idx = class_to_idx[item["true_class"]]

        cam_pred, _ = compute_gradcam(model, image_tensor, pred_idx)
        cam_true = cam_pred if pred_idx == true_idx else compute_gradcam(model, image_tensor, true_idx)[0]

        pred_at_threshold = binary_metrics(cam_pred, mask, cam_threshold)
        true_at_threshold = binary_metrics(cam_true, mask, cam_threshold)
        pred_rank = ranking_metrics(cam_pred, mask)
        true_rank = ranking_metrics(cam_true, mask)
        row: dict[str, Any] = {
            "filename": item["filename"],
            "true_class": item["true_class"],
            "predicted_class": class_names[pred_idx],
            "correct": bool(pred_idx == true_idx),
            "plane_code": item["plane_code"],
            "plane": {**PLANE_LABELS, "unknown": "unknown"}.get(item["plane_code"], "unknown"),
            "confidence_predicted": float(probs[pred_idx]),
            "score_true_class": float(probs[true_idx]),
            "mask_area_ratio": float(mask.mean()),
            "mask_positive_pixels": int(mask.sum()),
            "mask_values_before_binarization": ",".join(map(str, raw_mask_values)),
        }
        for key, value in pred_at_threshold.items():
            row[f"pred_cam_{key}_thr_{cam_threshold:.2f}"] = value
        for key, value in true_at_threshold.items():
            row[f"true_cam_{key}_thr_{cam_threshold:.2f}"] = value
        for key, value in pred_rank.items():
            row[f"pred_cam_{key}"] = value
        for key, value in true_rank.items():
            row[f"true_cam_{key}"] = value
        for fraction in (0.10, 0.20, 0.30):
            pm = metrics_at_top_fraction(cam_pred, mask, fraction)
            tm = metrics_at_top_fraction(cam_true, mask, fraction)
            row[f"pred_cam_dice_top_{int(fraction*100)}pct"] = pm["dice"]
            row[f"pred_cam_iou_top_{int(fraction*100)}pct"] = pm["iou"]
            row[f"true_cam_dice_top_{int(fraction*100)}pct"] = tm["dice"]
            row[f"true_cam_iou_top_{int(fraction*100)}pct"] = tm["iou"]

        rows.append(row)
        visual_data.append({
            **item,
            "display_image": np.asarray(display_image, dtype=np.float32) / 255.0,
            "mask": mask,
            "cam_pred": cam_pred,
            "cam_true": cam_true,
            "probabilities": probs,
            "predicted_class": class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "correct": bool(pred_idx == true_idx),
            "mask_area_ratio": float(mask.mean()),
        })
    return pd.DataFrame(rows), visual_data


def summarize_metrics(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    columns = [
        f"pred_cam_dice_thr_{threshold:.2f}",
        f"pred_cam_iou_thr_{threshold:.2f}",
        f"pred_cam_precision_thr_{threshold:.2f}",
        f"pred_cam_recall_thr_{threshold:.2f}",
        "pred_cam_pixel_roc_auc", "pred_cam_pixel_average_precision",
        f"true_cam_dice_thr_{threshold:.2f}",
        f"true_cam_iou_thr_{threshold:.2f}",
        "true_cam_pixel_roc_auc", "true_cam_pixel_average_precision",
    ]
    summaries = []
    groups = [("Geral", df)]
    groups.extend((f"Classe: {name}", part) for name, part in df.groupby("true_class"))
    groups.extend((f"Plano: {name}", part) for name, part in df.groupby("plane"))
    groups.extend((f"Resultado: {'correto' if key else 'incorreto'}", part)
                   for key, part in df.groupby("correct"))
    for label, part in groups:
        record: dict[str, Any] = {"group": label, "n_images": len(part)}
        for column in columns:
            if column in part:
                record[f"mean_{column}"] = float(part[column].mean())
        summaries.append(record)
    return pd.DataFrame(summaries)


def choose_visual_examples(visual_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Escolhe 1 exemplo correto por classe/plano, próximo da confiança mediana."""
    selected = []
    df = pd.DataFrame([
        {"idx": i, "true_class": x["true_class"], "plane_code": x["plane_code"],
         "correct": x["correct"], "confidence": x["confidence"]}
        for i, x in enumerate(visual_data)
    ])
    if df.empty:
        return selected
    for class_name in ("glioma", "meningioma", "pituitary"):
        for plane in ("ax", "co", "sa"):
            group = df[(df.true_class == class_name) & (df.plane_code == plane) & df.correct]
            if group.empty:
                continue
            median = float(group.confidence.median())
            choice = int(group.iloc[(group.confidence - median).abs().argmin()].idx)
            selected.append(visual_data[choice])
    return selected


def overlay_cam(ax, image: np.ndarray, cam: np.ndarray, title: str) -> None:
    ax.imshow(image)
    ax.imshow(cam, cmap="jet", alpha=0.42, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_example_panel(examples: list[dict[str, Any]], output: Path) -> None:
    if not examples:
        return
    fig, axes = plt.subplots(len(examples), 4, figsize=(13, max(4, 2.7 * len(examples))), squeeze=False)
    for r, item in enumerate(examples):
        axes[r, 0].imshow(item["display_image"])
        axes[r, 0].set_title(f"MRI — {item['true_class']} / {PLANE_LABELS.get(item['plane_code'], item['plane_code'])}", fontsize=8)
        axes[r, 1].imshow(item["mask"], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].set_title(f"Máscara de referência\nárea={item['mask_area_ratio']:.1%}", fontsize=8)
        overlay_cam(axes[r, 2], item["display_image"], item["cam_pred"],
                    f"Grad-CAM classe prevista\n{item['predicted_class']} ({item['confidence']:.1%})")
        overlay_cam(axes[r, 3], item["display_image"], item["cam_true"],
                    f"Grad-CAM classe real\n{item['true_class']}")
        axes[r, 0].axis("off")
        axes[r, 1].axis("off")
    fig.suptitle("Comparação Grad-CAM × máscara tumoral — exemplos corretos", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(fig)


def save_error_panel(visual_data: list[dict[str, Any]], output: Path, max_errors: int) -> None:
    errors = [x for x in visual_data if not x["correct"]]
    if not errors:
        return
    errors = errors[:max_errors]
    fig, axes = plt.subplots(len(errors), 4, figsize=(13, max(4, 2.8 * len(errors))), squeeze=False)
    for r, item in enumerate(errors):
        axes[r, 0].imshow(item["display_image"])
        axes[r, 0].set_title(f"Real: {item['true_class']}\nPlano: {item['plane_code']}", fontsize=8)
        axes[r, 1].imshow(item["mask"], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].set_title("Máscara tumoral", fontsize=8)
        overlay_cam(axes[r, 2], item["display_image"], item["cam_pred"],
                    f"CAM da previsão: {item['predicted_class']}\nscore={item['confidence']:.1%}")
        overlay_cam(axes[r, 3], item["display_image"], item["cam_true"],
                    f"CAM da classe real: {item['true_class']}")
        axes[r, 0].axis("off")
        axes[r, 1].axis("off")
    fig.suptitle("Erros de classificação: máscara × Grad-CAM da classe prevista e real", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(output, dpi=135, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=== COMPARAÇÃO GRAD-CAM × MÁSCARAS TUMORAIS — BRISC ===")
    print("Dispositivo:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Pasta de segmentação:", args.seg_root)

    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
    mean = tuple(checkpoint.get("normalization_mean", MEAN_DEFAULT))
    std = tuple(checkpoint.get("normalization_std", STD_DEFAULT))

    classification_test = DATA_ROOT / "test"
    if not classification_test.is_dir():
        raise FileNotFoundError(f"Pasta de teste de classificação não encontrada: {classification_test}")
    class_dataset = datasets.ImageFolder(classification_test)
    class_names = checkpoint.get("class_names", class_dataset.classes)
    if class_dataset.classes != class_names:
        raise ValueError(f"Ordem de classes divergente: {class_dataset.classes} vs {class_names}")

    pairs, pairing_df, pair_counts = pair_test_images(classification_test, args.seg_root, class_names)
    pairing_path = RESULTS_DIR / "mask_pairing_report.csv"
    pairing_df.to_csv(pairing_path, index=False)
    print("\n=== PAREAMENTO DAS IMAGENS E MÁSCARAS ===")
    for key, value in pair_counts.items():
        print(f"{key}: {value}")
    if pair_counts["matched_pairs"] == 0:
        raise RuntimeError(
            "Nenhum par imagem-máscara encontrado. Confira a pasta segmentation_task/test "
            "e se os nomes-base dos arquivos correspondem."
        )
    if pair_counts["missing_mask"] or pair_counts["missing_seg_image"] or pair_counts["dimension_mismatch"]:
        print("[AVISO] Há pares ausentes/incompatíveis. Consulte results/mask_pairing_report.csv.")

    model = load_model(checkpoint, class_names, device)
    print(f"\nPares válidos para análise: {len(pairs)}")
    print("Imagens no_tumor excluídas: não possuem máscara tumoral neste subconjunto.")
    metrics_df, visual_data = analyze_pairs(
        pairs, model, class_names, image_size, mean, std, device, args.cam_threshold
    )
    if metrics_df.empty:
        raise RuntimeError("Nenhuma imagem pôde ser avaliada após carregar as máscaras.")

    metrics_path = RESULTS_DIR / "gradcam_mask_metrics_per_image.csv"
    metrics_df.to_csv(metrics_path, index=False)
    summary_df = summarize_metrics(metrics_df, args.cam_threshold)
    summary_path = RESULTS_DIR / "gradcam_mask_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    examples = choose_visual_examples(visual_data)
    examples_path = RESULTS_DIR / "gradcam_mask_examples.png"
    errors_path = RESULTS_DIR / "gradcam_mask_errors.png"
    save_example_panel(examples, examples_path)
    save_error_panel(visual_data, errors_path, args.max_errors)

    primary_dice = f"pred_cam_dice_thr_{args.cam_threshold:.2f}"
    primary_iou = f"pred_cam_iou_thr_{args.cam_threshold:.2f}"
    primary_ap = "pred_cam_pixel_average_precision"
    primary_auc = "pred_cam_pixel_roc_auc"

    print("\n=== RESULTADO GERAL — CAM DA CLASSE PREVISTA × MÁSCARA ===")
    print(f"Imagens tumorais pareadas avaliadas: {len(metrics_df)}")
    print(f"Acurácia de classificação nessas imagens: {metrics_df['correct'].mean():.4f}")
    print(f"Dice médio (CAM >= {args.cam_threshold:.2f}): {metrics_df[primary_dice].mean():.4f}")
    print(f"IoU médio (CAM >= {args.cam_threshold:.2f}): {metrics_df[primary_iou].mean():.4f}")
    print(f"Pixel ROC-AUC médio por imagem: {metrics_df[primary_auc].mean():.4f}")
    print(f"Pixel Average Precision médio por imagem: {metrics_df[primary_ap].mean():.4f}")

    print("\n=== RESUMO POR CLASSE REAL (CAM DA CLASSE PREVISTA) ===")
    class_summary = metrics_df.groupby("true_class").agg(
        n_images=("filename", "count"),
        correct_rate=("correct", "mean"),
        dice=(primary_dice, "mean"),
        iou=(primary_iou, "mean"),
        pixel_ap=(primary_ap, "mean"),
    )
    print(class_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n=== RESUMO POR PLANO (CAM DA CLASSE PREVISTA) ===")
    plane_summary = metrics_df.groupby("plane").agg(
        n_images=("filename", "count"),
        correct_rate=("correct", "mean"),
        dice=(primary_dice, "mean"),
        iou=(primary_iou, "mean"),
        pixel_ap=(primary_ap, "mean"),
    )
    print(plane_summary.to_string(float_format=lambda x: f"{x:.4f}"))

    summary_json = {
        "checkpoint": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "segmentation_root": str(args.seg_root),
        "mask_pairing": pair_counts,
        "evaluated_pairs": int(len(metrics_df)),
        "cam_threshold": float(args.cam_threshold),
        "cam_threshold_note": "Limiar aplicado à CAM normalizada individualmente para Dice/IoU.",
        "metrics_note": (
            "Grad-CAM é mapa de saliência de baixa resolução, não máscara de segmentação. "
            "Dice/IoU e métricas pixel a pixel quantificam apenas alinhamento espacial aproximado. "
            "A CAM da classe prevista explica a decisão efetiva; a CAM da classe real é uma análise comparativa."
        ),
        "no_tumor_note": (
            "Imagens no_tumor foram excluídas da comparação de sobreposição porque o subconjunto "
            "de segmentação do BRISC fornece máscaras para as três classes tumorais, não máscaras vazias para todos os no_tumor."
        ),
        "overall_predicted_class_cam": {
            "dice": float(metrics_df[primary_dice].mean()),
            "iou": float(metrics_df[primary_iou].mean()),
            "pixel_roc_auc": float(metrics_df[primary_auc].mean()),
            "pixel_average_precision": float(metrics_df[primary_ap].mean()),
        },
    }
    json_path = RESULTS_DIR / "gradcam_mask_summary.json"
    json_path.write_text(json.dumps(summary_json, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== ARQUIVOS GERADOS ===")
    for path in [pairing_path, metrics_path, summary_path, json_path, examples_path, errors_path]:
        if path.exists():
            print(path)
    print("\nConcluído. Nenhum peso foi treinado ou atualizado.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        raise
