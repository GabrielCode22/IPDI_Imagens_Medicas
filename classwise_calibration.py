#!/usr/bin/env python3
"""Investigação de calibração por categoria e plano anatômico para BRISC.

- Ajusta uma temperatura escalar APENAS no conjunto de validação.
- Avalia o conjunto oficial de teste uma única vez, sem ajustar parâmetros nele.
- Calcula calibração one-vs-rest por categoria (ECE e Brier binário).
- Calcula calibração da confiança entre previsões atribuídas a cada categoria.
- Resume ECE one-vs-rest por plano (axial, coronal e sagital).
- Gera curvas de confiabilidade e tabelas CSV/JSON.

Execute na raiz do projeto:
    python classwise_calibration.py
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
PLANE_PATTERN = re.compile(r"_(ax|co|sa)(?:_|\.)", re.IGNORECASE)
PLANE_NAMES = {"ax": "Axial", "co": "Coronal", "sa": "Sagital"}


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}. Execute python train.py primeiro.")
    try:
        data = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict) or "model_state_dict" not in data:
        raise ValueError("Checkpoint inválido: campo model_state_dict ausente.")
    return data


def collect_logits(model: nn.Module, loader: torch.utils.data.DataLoader,
                   device: torch.device, description: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_out: list[np.ndarray] = []
    targets_out: list[np.ndarray] = []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for images, targets in tqdm(loader, desc=description):
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
            logits_out.append(logits.float().cpu().numpy())
            targets_out.append(targets.cpu().numpy())
    if not logits_out:
        raise RuntimeError(f"Nenhum exemplo carregado: {description}")
    return np.concatenate(logits_out), np.concatenate(targets_out).astype(int)


def fit_temperature(logits_np: np.ndarray, targets_np: np.ndarray) -> tuple[float, dict[str, Any]]:
    logits = torch.tensor(logits_np, dtype=torch.float64)
    targets = torch.tensor(targets_np, dtype=torch.long)

    def objective(log_t: float) -> float:
        t = float(np.exp(log_t))
        with torch.no_grad():
            return float(F.cross_entropy(logits / t, targets).item())

    result = minimize_scalar(
        objective,
        bounds=(np.log(MIN_TEMPERATURE), np.log(MAX_TEMPERATURE)),
        method="bounded",
        options={"xatol": 1e-8, "maxiter": 500},
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"Falha ao ajustar temperatura na validação: {result.message}")
    return float(np.exp(result.x)), {
        "success": bool(result.success),
        "validation_nll_at_optimum": float(result.fun),
        "iterations": int(result.nfev),
    }


def ece_binary(y_binary: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> tuple[float, list[dict[str, float | int]]]:
    y_binary = np.asarray(y_binary, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.minimum(np.digitize(p, edges[1:-1], right=False), n_bins - 1)
    ece = 0.0
    rows: list[dict[str, float | int]] = []
    for b in range(n_bins):
        mask = ids == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_p = float(p[mask].mean())
        frac_pos = float(y_binary[mask].mean())
        ece += n / len(p) * abs(frac_pos - mean_p)
        rows.append({"bin": b + 1, "count": n, "mean_probability": mean_p,
                     "observed_frequency": frac_pos})
    return float(ece), rows


def top_label_ece(y: np.ndarray, probs: np.ndarray, n_bins: int = N_BINS) -> tuple[float, list[dict[str, float | int]]]:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    return ece_binary((pred == y).astype(float), conf, n_bins)


def brier_binary(y_binary: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y_binary, dtype=np.float64)) ** 2))


def multiclass_brier(y: np.ndarray, probs: np.ndarray) -> float:
    onehot = np.eye(probs.shape[1], dtype=np.float64)[y]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def parse_plane(path: str) -> str:
    match = PLANE_PATTERN.search(Path(path).name)
    if not match:
        return "Desconhecido"
    return PLANE_NAMES[match.group(1).lower()]


def get_test_paths(test_loader: torch.utils.data.DataLoader, n: int) -> list[str]:
    dataset = test_loader.dataset
    samples = getattr(dataset, "samples", None)
    if samples is None or len(samples) != n:
        raise RuntimeError("Não foi possível alinhar caminhos das imagens à ordem do teste.")
    return [str(item[0]) for item in samples]


def make_class_tables(y: np.ndarray, before: np.ndarray, after: np.ndarray,
                      class_names: list[str]) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    pred_before = before.argmax(axis=1)
    pred_after = after.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    reliability: list[dict[str, Any]] = []

    for k, name in enumerate(class_names):
        binary = (y == k).astype(float)
        ece_b, bins_b = ece_binary(binary, before[:, k])
        ece_a, bins_a = ece_binary(binary, after[:, k])
        onehot_brier_before = brier_binary(binary, before[:, k])
        onehot_brier_after = brier_binary(binary, after[:, k])
        selected_before = pred_before == k
        selected_after = pred_after == k
        correct_before = (pred_before == y)
        correct_after = (pred_after == y)

        rows.append({
            "class": name,
            "positive_support": int(binary.sum()),
            "ovr_ece_before": ece_b,
            "ovr_ece_after": ece_a,
            "ovr_brier_before": onehot_brier_before,
            "ovr_brier_after": onehot_brier_after,
            "predicted_count": int(selected_after.sum()),
            "precision_when_predicted": float(correct_after[selected_after].mean()) if selected_after.any() else float("nan"),
            "mean_confidence_when_predicted": float(after[selected_after, k].mean()) if selected_after.any() else float("nan"),
            "predicted_class_calibration_gap": float(after[selected_after, k].mean() - correct_after[selected_after].mean()) if selected_after.any() else float("nan"),
            "errors_when_predicted": int((selected_after & ~correct_after).sum()),
            "mean_confidence_on_false_positives": float(after[selected_after & ~correct_after, k].mean()) if (selected_after & ~correct_after).any() else float("nan"),
            "true_class_recall": float((pred_after[y == k] == k).mean()) if np.any(y == k) else float("nan"),
        })
        reliability.append({"class": name, "before": bins_b, "after": bins_a})

    # Calibração da confiança máxima condicionada à classe que o modelo previu.
    pred_rows: list[dict[str, Any]] = []
    conf_after = after.max(axis=1)
    for k, name in enumerate(class_names):
        subset = pred_after == k
        count = int(subset.sum())
        if count:
            ece_cond, _ = ece_binary((pred_after[subset] == y[subset]).astype(float), conf_after[subset])
            mean_conf = float(conf_after[subset].mean())
            accuracy_cond = float((pred_after[subset] == y[subset]).mean())
        else:
            ece_cond, mean_conf, accuracy_cond = float("nan"), float("nan"), float("nan")
        pred_rows.append({"predicted_class": name, "n_predictions": count,
                          "accuracy_among_predictions": accuracy_cond,
                          "mean_top_confidence": mean_conf,
                          "confidence_accuracy_gap": mean_conf - accuracy_cond if count else float("nan"),
                          "conditional_top_label_ece": ece_cond})
    return pd.DataFrame(rows), reliability, pd.DataFrame(pred_rows)


def make_orientation_table(y: np.ndarray, before: np.ndarray, after: np.ndarray,
                           paths: list[str], class_names: list[str]) -> pd.DataFrame:
    planes = np.array([parse_plane(p) for p in paths])
    rows: list[dict[str, Any]] = []
    for plane in ["Axial", "Coronal", "Sagital", "Desconhecido"]:
        mask_plane = planes == plane
        if not mask_plane.any():
            continue
        for k, name in enumerate(class_names):
            sub = mask_plane
            binary = (y[sub] == k).astype(float)
            if not len(binary):
                continue
            ece_b, _ = ece_binary(binary, before[sub, k])
            ece_a, _ = ece_binary(binary, after[sub, k])
            rows.append({"plane": plane, "class": name, "n_images": int(sub.sum()),
                         "class_positive_support": int(binary.sum()),
                         "ovr_ece_before": ece_b, "ovr_ece_after": ece_a,
                         "ovr_brier_before": brier_binary(binary, before[sub, k]),
                         "ovr_brier_after": brier_binary(binary, after[sub, k])})
    return pd.DataFrame(rows)


def plot_class_reliability(reliability: list[dict[str, Any]], out: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, item in zip(axes.flat, reliability):
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibração perfeita")
        for key, label, marker in [("before", "Antes", "o"), ("after", "Depois", "s")]:
            bins = item[key]
            if bins:
                ax.plot([r["mean_probability"] for r in bins],
                        [r["observed_frequency"] for r in bins],
                        marker=marker, linewidth=1.7, label=label)
        ax.set_title(item["class"])
        ax.set_xlabel("Probabilidade prevista para a classe")
        ax.set_ylabel("Frequência observada da classe")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Calibração one-vs-rest por categoria — teste BRISC", fontsize=14)
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_ece(class_df: pd.DataFrame, out: Path) -> None:
    x = np.arange(len(class_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, class_df["ovr_ece_before"], width, label="Antes")
    ax.bar(x + width / 2, class_df["ovr_ece_after"], width, label="Depois")
    ax.set_xticks(x, class_df["class"], rotation=15, ha="right")
    ax.set_ylabel("ECE one-vs-rest (menor é melhor)")
    ax.set_title("Erro de calibração por categoria")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== CALIBRAÇÃO POR CATEGORIA — RESNET18 / BRISC ===")
    print("Dispositivo:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("Temperatura global ajustada somente na validação.")

    _train_loader, val_loader, test_loader, pipeline_classes = build_dataloaders()
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    class_names = checkpoint.get("class_names", pipeline_classes)
    if class_names != pipeline_classes:
        raise ValueError(f"Ordem de classes incompatível: checkpoint={class_names}; pipeline={pipeline_classes}")

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    val_logits, val_y = collect_logits(model, val_loader, device, "Logits de validação")
    test_logits, test_y = collect_logits(model, test_loader, device, "Logits do teste oficial")
    temperature, opt_info = fit_temperature(val_logits, val_y)
    before = softmax(test_logits, axis=1)
    after = softmax(test_logits / temperature, axis=1)

    if not np.array_equal(before.argmax(axis=1), after.argmax(axis=1)):
        raise RuntimeError("A temperatura alterou o argmax, o que não deveria acontecer.")

    paths = get_test_paths(test_loader, len(test_y))
    class_df, reliability, predicted_df = make_class_tables(test_y, before, after, class_names)
    orient_df = make_orientation_table(test_y, before, after, paths, class_names)

    top_ece_before, _ = top_label_ece(test_y, before)
    top_ece_after, _ = top_label_ece(test_y, after)
    summary = {
        "method": "global_scalar_temperature_scaling",
        "temperature": temperature,
        "temperature_fit_split": "validation_only",
        "test_size": int(len(test_y)),
        "overall_test": {
            "accuracy": float(accuracy_score(test_y, after.argmax(axis=1))),
            "macro_f1": float(f1_score(test_y, after.argmax(axis=1), average="macro", zero_division=0)),
            "top_label_ece_before": top_ece_before,
            "top_label_ece_after": top_ece_after,
            "macro_ovr_ece_before": float(class_df["ovr_ece_before"].mean()),
            "macro_ovr_ece_after": float(class_df["ovr_ece_after"].mean()),
            "macro_ovr_brier_before": float(class_df["ovr_brier_before"].mean()),
            "macro_ovr_brier_after": float(class_df["ovr_brier_after"].mean()),
            "multiclass_brier_before": multiclass_brier(test_y, before),
            "multiclass_brier_after": multiclass_brier(test_y, after),
            "nll_before": float(log_loss(test_y, before, labels=list(range(len(class_names))))),
            "nll_after": float(log_loss(test_y, after, labels=list(range(len(class_names))))),
        },
        "classwise_ovr_definition": "For each class k, evaluate p(y=k) against binary indicator [y=k] using 15 equal-width bins.",
        "conditional_top_label_definition": "Among samples whose argmax is class k, compare mean max probability with empirical accuracy; conditional ECE uses 15 bins.",
        "orientation_note": "Orientation parsed from filename suffix _ax_, _co_, or _sa_; orientation metrics are descriptive subgroups, not separately calibrated.",
        "optimizer": opt_info,
        "classwise_results": class_df.to_dict(orient="records"),
        "conditional_top_label_results": predicted_df.to_dict(orient="records"),
        "limitations": [
            "All category-wise results are evaluated on the official test partition; no per-class temperatures are fitted on test.",
            "ECE depends on binning and can be noisy in subgroups with limited support.",
            "A global scalar temperature preserves argmax and ranking; it cannot correct class-specific ranking errors.",
            "The BRISC split is image-based and does not establish patient-independent evaluation."
        ],
    }

    class_csv = RESULTS_DIR / "classwise_calibration_metrics.csv"
    conditional_csv = RESULTS_DIR / "classwise_prediction_confidence.csv"
    orientation_csv = RESULTS_DIR / "classwise_calibration_by_plane.csv"
    reliability_csv = RESULTS_DIR / "classwise_reliability_bins.csv"
    class_df.to_csv(class_csv, index=False)
    predicted_df.to_csv(conditional_csv, index=False)
    orient_df.to_csv(orientation_csv, index=False)
    reliability_rows = []
    for item in reliability:
        for stage in ("before", "after"):
            for bin_row in item[stage]:
                reliability_rows.append({"class": item["class"], "stage": stage, **bin_row})
    pd.DataFrame(reliability_rows).to_csv(reliability_csv, index=False)

    pred = after.argmax(axis=1)
    rows = []
    for i, (path, true_id, pred_id, p0, p1) in enumerate(zip(paths, test_y, pred, before, after)):
        row: dict[str, Any] = {
            "index": i, "image_path": path, "plane": parse_plane(path),
            "true_class": class_names[int(true_id)], "predicted_class": class_names[int(pred_id)],
            "correct": bool(true_id == pred_id),
            "top_confidence_before": float(p0.max()), "top_confidence_after": float(p1.max()),
        }
        for k, name in enumerate(class_names):
            row[f"prob_before_{name}"] = float(p0[k])
            row[f"prob_after_{name}"] = float(p1[k])
        rows.append(row)
    predictions_csv = RESULTS_DIR / "classwise_calibration_test_predictions.csv"
    pd.DataFrame(rows).to_csv(predictions_csv, index=False)

    curves_png = RESULTS_DIR / "classwise_reliability_curves.png"
    ece_png = RESULTS_DIR / "classwise_ece_comparison.png"
    plot_class_reliability(reliability, curves_png)
    plot_class_ece(class_df, ece_png)

    json_path = RESULTS_DIR / "classwise_calibration_summary.json"
    def json_safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None
        if isinstance(value, (np.bool_,)):
            return bool(value)
        return value

    json_path.write_text(
        json.dumps(json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    report_path = RESULTS_DIR / "classwise_calibration_report.txt"
    lines = [
        "CALIBRAÇÃO POR CATEGORIA — RESNET18 / BRISC",
        f"Temperatura global ajustada na validação: {temperature:.6f}",
        "\nRESULTADOS ONE-VS-REST POR CLASSE (teste oficial)",
        class_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "\nCONFIANÇA CONDICIONADA À CLASSE PREVISTA",
        predicted_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "\nRESULTADOS ONE-VS-REST POR PLANO (subgrupos descritivos)",
        orient_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
        "\nObservações:",
        "- A temperatura foi ajustada somente na validação, não no teste.",
        "- O ECE one-vs-rest compara p(classe) com a frequência real daquela classe.",
        "- A confiança condicionada à classe prevista é uma análise complementar.",
        "- ECE depende do número e dos limites dos bins; observe também os suportes.",
        "- A calibração global não corrige erros de ranking nem substitui validação por paciente.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nTemperatura ajustada na validação: {temperature:.6f}")
    print("\n=== CALIBRAÇÃO ONE-VS-REST POR CATEGORIA — TESTE ===")
    print(class_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== CONFIANÇA ENTRE PREVISÕES DE CADA CATEGORIA ===")
    print(predicted_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== ECE ONE-VS-REST POR ORIENTAÇÃO E CATEGORIA ===")
    print(orient_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== ARQUIVOS GERADOS ===")
    for path in [class_csv, conditional_csv, orientation_csv, reliability_csv, predictions_csv, curves_png, ece_png, json_path, report_path]:
        print(path)
    print("\nConcluído. Nenhum peso foi atualizado; nenhuma temperatura por classe foi ajustada no teste.")


if __name__ == "__main__":
    main()
