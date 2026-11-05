#!/usr/bin/env python3
"""Analisa erros pareados da ResNet18 original e filtrada no teste BRISC.

Executar na raiz do projeto:
    python analyze_model_errors.py

Somente leitura: nao treina, nao modifica checkpoints e nao altera imagens.
Cruza erros com candidatos pHash e duplicatas exatas encontrados na auditoria.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "model_comparison"
AUDIT = ROOT / "results" / "leakage_audit"
OUT = RESULTS / "error_analysis"
CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def norm_path(value: object) -> str:
    return str(Path(str(value)).expanduser().resolve())


def split_group_paths(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [norm_path(x.strip()) for x in str(value).split("||") if x.strip()]


def plane_from_path(path: str) -> str:
    stem = Path(path).stem.lower()
    parts = stem.split("_")
    if len(parts) >= 2:
        code = parts[-2]
        return {"ax": "axial", "co": "coronal", "sa": "sagittal"}.get(code, "unknown")
    return "unknown"


def safe_read_csv(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo necessário não encontrado: {path}")
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {sorted(missing)}")
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pair_path = RESULTS / "paired_test_predictions.csv"
    cand_path = AUDIT / "train_candidates_to_review.csv"
    near_path = AUDIT / "near_duplicate_pairs.csv"
    exact_path = AUDIT / "exact_duplicate_groups.csv"

    df = safe_read_csv(pair_path, {
        "image_path", "true_class", "original_prediction", "filtered_prediction",
        "original_correct", "filtered_correct", "prediction_changed",
        "original_confidence", "filtered_confidence",
    }).copy()
    df["image_path_norm"] = df["image_path"].map(norm_path)
    df["plane"] = df["image_path_norm"].map(plane_from_path)
    for col in ("original_correct", "filtered_correct", "prediction_changed"):
        if df[col].dtype != bool:
            df[col] = df[col].astype(str).str.lower().isin({"true", "1", "yes"})

    # Carrega os candidatos transitivos e relaciona cada imagem de teste aos
    # caminhos de treinamento presentes no mesmo componente pHash/SHA-256.
    candidates = safe_read_csv(cand_path, {"train_path", "test_paths_in_component", "class", "plane"})
    membership_path = AUDIT / "candidate_membership.csv"
    membership_df = safe_read_csv(membership_path, {"train_path", "original_internal_split"})
    train_membership = {
        norm_path(row.train_path): str(row.original_internal_split)
        for row in membership_df.itertuples(index=False)
    }
    test_to_candidate_train: dict[str, set[str]] = defaultdict(set)
    for row in candidates.itertuples(index=False):
        train_path = norm_path(row.train_path)
        for test_path in split_group_paths(row.test_paths_in_component):
            test_to_candidate_train[test_path].add(train_path)

    # Correspondencias diretas pHash train-test.
    near = safe_read_csv(near_path, {"path_a", "split_a", "path_b", "split_b", "hamming_distance"})
    direct_matches: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in near.itertuples(index=False):
        sa, sb = str(row.split_a), str(row.split_b)
        if {sa, sb} == {"train", "test"}:
            if sa == "test":
                test_path, train_path = norm_path(row.path_a), norm_path(row.path_b)
            else:
                test_path, train_path = norm_path(row.path_b), norm_path(row.path_a)
            direct_matches[test_path].append((train_path, int(row.hamming_distance)))

    # Duplicatas exatas SHA-256 entre treino e teste.
    exact = safe_read_csv(exact_path, {"splits", "paths", "classes", "planes"})
    exact_train_by_test: dict[str, set[str]] = defaultdict(set)
    for row in exact.itertuples(index=False):
        splits = set(str(row.splits).split(";"))
        if not {"train", "test"}.issubset(splits):
            continue
        paths = split_group_paths(row.paths)
        train_paths = [p for p in paths if "/train/" in p.replace("\\", "/")]
        test_paths = [p for p in paths if "/test/" in p.replace("\\", "/")]
        for test_path in test_paths:
            exact_train_by_test[test_path].update(train_paths)

    # Adiciona sinalizadores de auditoria a cada previsão do teste.
    df["n_direct_train_phash_matches"] = df["image_path_norm"].map(lambda p: len(direct_matches.get(p, [])))
    df["min_train_phash_hamming"] = df["image_path_norm"].map(
        lambda p: min((d for _, d in direct_matches.get(p, [])), default=np.nan)
    )
    df["n_train_candidates_in_transitive_component"] = df["image_path_norm"].map(
        lambda p: len(test_to_candidate_train.get(p, set()))
    )
    df["n_transitive_candidates_from_original_fit"] = df["image_path_norm"].map(
        lambda p: sum(train_membership.get(t, "") == "fit_train_4000" for t in test_to_candidate_train.get(p, set()))
    )
    df["n_transitive_candidates_from_original_validation"] = df["image_path_norm"].map(
        lambda p: sum(train_membership.get(t, "") == "validation_1000" for t in test_to_candidate_train.get(p, set()))
    )
    df["n_direct_fit_train_matches"] = df["image_path_norm"].map(
        lambda p: sum(train_membership.get(t, "") == "fit_train_4000" for t, _ in direct_matches.get(p, []))
    )
    df["n_exact_train_copies"] = df["image_path_norm"].map(lambda p: len(exact_train_by_test.get(p, set())))
    df["n_exact_fit_train_copies"] = df["image_path_norm"].map(
        lambda p: sum(train_membership.get(t, "") == "fit_train_4000" for t in exact_train_by_test.get(p, set()))
    )
    df["has_direct_train_phash_match"] = df["n_direct_train_phash_matches"] > 0
    df["has_transitive_train_candidate"] = df["n_train_candidates_in_transitive_component"] > 0
    df["has_exact_train_copy"] = df["n_exact_train_copies"] > 0
    df["has_exact_fit_train_copy"] = df["n_exact_fit_train_copies"] > 0

    df["original_error"] = ~df["original_correct"]
    df["filtered_error"] = ~df["filtered_correct"]
    conditions = [
        df["original_correct"] & df["filtered_correct"],
        df["original_correct"] & ~df["filtered_correct"],
        ~df["original_correct"] & df["filtered_correct"],
        ~df["original_correct"] & ~df["filtered_correct"],
    ]
    labels = ["ambos_corretos", "somente_filtrado_errou", "somente_original_errou", "ambos_erraram"]
    df["comparison_group"] = np.select(conditions, labels, default="indefinido")

    # Probabilidade da classe verdadeira e margem entre maior e segunda maior
    # probabilidade para cada modelo, caso as colunas existam.
    for model in ("original", "filtered"):
        prob_cols = [f"{model}_prob_{c}" for c in CLASSES]
        if set(prob_cols).issubset(df.columns):
            true_col = df["true_class"].map(lambda c: f"{model}_prob_{c}")
            df[f"{model}_true_class_probability"] = [float(df.iloc[i][col]) for i, col in enumerate(true_col)]
            arr = df[prob_cols].to_numpy(dtype=float)
            sorted_probs = np.sort(arr, axis=1)
            df[f"{model}_top2_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]

    # Exporta erros do modelo filtrado e os casos de discordância.
    filtered_errors = df[df["filtered_error"]].copy()
    disagreements = df[df["prediction_changed"]].copy()
    df.to_csv(OUT / "all_test_predictions_with_audit_flags.csv", index=False)
    filtered_errors.to_csv(OUT / "filtered_model_errors.csv", index=False)
    disagreements.to_csv(OUT / "model_disagreements.csv", index=False)
    df[df["comparison_group"] == "somente_filtrado_errou"].to_csv(
        OUT / "errors_only_filtered.csv", index=False
    )
    df[df["comparison_group"] == "somente_original_errou"].to_csv(
        OUT / "errors_only_original.csv", index=False
    )
    df[df["comparison_group"] == "ambos_erraram"].to_csv(
        OUT / "errors_both_models.csv", index=False
    )

    # Contagens de confusão e erros por orientação.
    filtered_confusions = (
        filtered_errors.groupby(["true_class", "filtered_prediction"], dropna=False)
        .size().rename("n").reset_index().sort_values("n", ascending=False)
    )
    filtered_confusions.to_csv(OUT / "filtered_error_confusions.csv", index=False)
    df["filtered_error_with_direct_phash"] = df["filtered_error"] & df["has_direct_train_phash_match"]
    df["filtered_error_with_transitive_candidate"] = df["filtered_error"] & df["has_transitive_train_candidate"]
    by_plane = (
        df.groupby("plane").agg(
            n_images=("image_path", "size"),
            original_errors=("original_error", "sum"),
            filtered_errors=("filtered_error", "sum"),
            changed_predictions=("prediction_changed", "sum"),
            filtered_errors_with_direct_phash=("filtered_error_with_direct_phash", "sum"),
            filtered_errors_with_transitive_candidate=("filtered_error_with_transitive_candidate", "sum"),
        ).reset_index()
    )
    by_plane.to_csv(OUT / "errors_by_plane.csv", index=False)

    # Resumo de associação entre erros do filtrado e sinais de similaridade.
    def flag_summary(sub: pd.DataFrame, flag: str) -> dict[str, int | float]:
        n = int(sub[flag].sum())
        return {"n_images": int(len(sub)), "n_flagged": n,
                "fraction_flagged": float(n / len(sub)) if len(sub) else float("nan")}

    summary = {
        "n_test_images": int(len(df)),
        "n_original_errors": int(df["original_error"].sum()),
        "n_filtered_errors": int(df["filtered_error"].sum()),
        "n_both_correct": int((df["comparison_group"] == "ambos_corretos").sum()),
        "n_only_filtered_wrong": int((df["comparison_group"] == "somente_filtrado_errou").sum()),
        "n_only_original_wrong": int((df["comparison_group"] == "somente_original_errou").sum()),
        "n_both_wrong": int((df["comparison_group"] == "ambos_erraram").sum()),
        "n_prediction_changes": int(df["prediction_changed"].sum()),
        "filtered_error_true_to_predicted": {
            f"{r.true_class} -> {r.filtered_prediction}": int(r.n)
            for r in filtered_confusions.itertuples(index=False)
        },
        "filtered_errors_with_direct_train_phash_match": flag_summary(filtered_errors, "has_direct_train_phash_match"),
        "filtered_errors_with_transitive_train_candidate": flag_summary(filtered_errors, "has_transitive_train_candidate"),
        "filtered_errors_with_exact_train_copy": flag_summary(filtered_errors, "has_exact_train_copy"),
        "filtered_errors_with_exact_copy_in_original_fit": flag_summary(filtered_errors, "has_exact_fit_train_copy"),
        "filtered_errors_with_direct_match_to_original_fit": {
            "n_images": int(len(filtered_errors)),
            "n_flagged": int((filtered_errors["n_direct_fit_train_matches"] > 0).sum()),
        },
        "filtered_errors_with_transitive_candidates_from_original_fit": {
            "n_images": int(len(filtered_errors)),
            "n_flagged": int((filtered_errors["n_transitive_candidates_from_original_fit"] > 0).sum()),
        },
        "all_test_with_direct_train_phash_match": flag_summary(df, "has_direct_train_phash_match"),
        "all_test_with_transitive_train_candidate": flag_summary(df, "has_transitive_train_candidate"),
        "all_test_with_exact_train_copy": flag_summary(df, "has_exact_train_copy"),
        "caveats": [
            "pHash indica similaridade visual e nao identidade de paciente.",
            "Associacao entre erro e sinal de similaridade nao prova causalidade.",
            "Os 120 casos test-test relacionados podem gerar dependencia interna no conjunto de teste.",
            "Resultados sao descritivos para o teste BRISC oficial, nao validacao clinica independente.",
        ],
    }
    (OUT / "error_analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Grafico dos erros filtrados por categoria verdadeira e classe prevista.
    matrix = pd.crosstab(filtered_errors["true_class"], filtered_errors["filtered_prediction"])
    matrix = matrix.reindex(index=CLASSES, columns=CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix.to_numpy(), cmap="Reds")
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=25, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Classe prevista pelo modelo filtrado")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title("Erros do modelo filtrado — teste oficial")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(int(matrix.iloc[i, j])), ha="center", va="center")
    fig.colorbar(im, ax=ax, label="Quantidade de erros")
    fig.tight_layout()
    fig.savefig(OUT / "filtered_error_confusion.png", dpi=160)
    plt.close(fig)

    # Gráfico: proporção de erros filtrados com sinal de similaridade vs todas as imagens.
    flags = [
        ("Ligação pHash direta", "has_direct_train_phash_match"),
        ("Candidato transitivo", "has_transitive_train_candidate"),
        ("Cópia exata no treino", "has_exact_train_copy"),
    ]
    vals_errors = [100 * filtered_errors[f].mean() if len(filtered_errors) else 0 for _, f in flags]
    vals_all = [100 * df[f].mean() for _, f in flags]
    x = np.arange(len(flags))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, vals_all, width, label="Todas as imagens de teste")
    ax.bar(x + width / 2, vals_errors, width, label="Erros do modelo filtrado")
    ax.set_xticks(x, [name for name, _ in flags], rotation=12, ha="right")
    ax.set_ylabel("Imagens (%)")
    ax.set_title("Sinais de similaridade nas imagens de teste")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "similarity_flags_error_comparison.png", dpi=160)
    plt.close(fig)

    report = [
        "=== ANÁLISE DOS ERROS — RESNET18 ORIGINAL × FILTRADA ===",
        f"Imagens de teste: {len(df)}",
        f"Erros original: {summary['n_original_errors']}",
        f"Erros filtrado: {summary['n_filtered_errors']}",
        f"Ambos corretos: {summary['n_both_correct']}",
        f"Somente filtrado errou: {summary['n_only_filtered_wrong']}",
        f"Somente original errou: {summary['n_only_original_wrong']}",
        f"Ambos erraram: {summary['n_both_wrong']}",
        f"Previsões diferentes: {summary['n_prediction_changes']}",
        "",
        "Erros filtrado por classe real -> classe prevista:",
        filtered_confusions.to_string(index=False) if len(filtered_confusions) else "Nenhum erro.",
        "",
        "Sinais de similaridade entre erros do filtrado:",
        f"Ligação pHash direta: {summary['filtered_errors_with_direct_train_phash_match']}",
        f"Candidato transitivo: {summary['filtered_errors_with_transitive_train_candidate']}",
        f"Cópia exata no treino original: {summary['filtered_errors_with_exact_train_copy']}",
        f"Correspondência direta com treino efetivo original: {summary['filtered_errors_with_direct_match_to_original_fit']}",
        f"Candidato transitivo oriundo do treino efetivo original: {summary['filtered_errors_with_transitive_candidates_from_original_fit']}",
        "",
        "Erros por orientação:",
        by_plane.to_string(index=False),
        "",
        "Ressalvas: pHash nao identifica pacientes; associação não prova causalidade.",
    ]
    (OUT / "error_analysis_report.txt").write_text("\n".join(report), encoding="utf-8")

    print("=== ANÁLISE DOS ERROS — RESNET18 ORIGINAL × FILTRADA ===")
    print(f"Imagens de teste: {len(df)}")
    print(f"Erros original: {summary['n_original_errors']} | erros filtrado: {summary['n_filtered_errors']}")
    print(f"Ambos corretos: {summary['n_both_correct']}")
    print(f"Somente filtrado errou: {summary['n_only_filtered_wrong']}")
    print(f"Somente original errou: {summary['n_only_original_wrong']}")
    print(f"Ambos erraram: {summary['n_both_wrong']}")
    print(f"Previsões diferentes: {summary['n_prediction_changes']}")
    print("\n=== ERROS DO MODELO FILTRADO — CLASSE REAL -> PREVISTA ===")
    print(filtered_confusions.to_string(index=False) if len(filtered_confusions) else "Nenhum erro.")
    print("\n=== SINAIS DE SIMILARIDADE ENTRE ERROS DO FILTRADO ===")
    for key in (
        "filtered_errors_with_direct_train_phash_match",
        "filtered_errors_with_transitive_train_candidate",
        "filtered_errors_with_exact_train_copy",
        "filtered_errors_with_exact_copy_in_original_fit",
        "filtered_errors_with_direct_match_to_original_fit",
        "filtered_errors_with_transitive_candidates_from_original_fit",
    ):
        print(f"{key}: {summary[key]}")
    print("\n=== ERROS POR ORIENTAÇÃO ===")
    print(by_plane.to_string(index=False))
    print("\n=== ARQUIVOS GERADOS ===")
    for p in sorted(OUT.iterdir()):
        print(p)
    print("\nAnálise concluída. Nenhum modelo, checkpoint ou imagem foi modificado.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
