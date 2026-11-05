#!/usr/bin/env python3
"""Reconstitui a divisão original treino/validação e detalha candidatos de leakage.

Somente leitura: não move, apaga ou altera imagens, pesos ou resultados existentes.
Reproduz exatamente a divisão de data_pipeline.py:
  train_test_split(indices, test_size=0.20, random_state=42, stratify=labels)
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import datasets

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "brisc2025" / "classification_task"
AUDIT_DIR = ROOT / "results" / "leakage_audit"
OUT_DIR = ROOT / "results" / "leakage_audit"
SEED = 42
VAL_FRACTION = 0.20


def norm_path(value: str | Path) -> str:
    return str(Path(str(value)).expanduser().resolve())


def main() -> None:
    required = [
        DATA_ROOT / "train",
        DATA_ROOT / "test",
        AUDIT_DIR / "train_candidates_to_review.csv",
        AUDIT_DIR / "near_duplicate_pairs.csv",
        AUDIT_DIR / "exact_duplicate_groups.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Arquivos/pastas ausentes:\n" + "\n".join(missing))

    ds = datasets.ImageFolder(DATA_ROOT / "train")
    idx = list(range(len(ds)))
    train_idx, val_idx = train_test_split(
        idx,
        test_size=VAL_FRACTION,
        random_state=SEED,
        stratify=ds.targets,
    )

    paths = [norm_path(path) for path, _ in ds.samples]
    membership = {}
    for i in train_idx:
        membership[paths[i]] = "fit_train_4000"
    for i in val_idx:
        membership[paths[i]] = "validation_1000"

    candidates = pd.read_csv(AUDIT_DIR / "train_candidates_to_review.csv")
    candidates["train_path_norm"] = candidates["train_path"].map(norm_path)
    candidates["original_internal_split"] = candidates["train_path_norm"].map(membership).fillna("NOT_FOUND_IN_SPLIT")

    exact = pd.read_csv(AUDIT_DIR / "exact_duplicate_groups.csv")
    # Grupos exatos cujos nomes de arquivo indicam mais de um plano.
    # Isso é uma pista de qualidade dos metadados, não prova de orientação incorreta.
    exact_multi_plane = exact[exact["planes"].astype(str).str.contains(";", regex=False)].copy()
    exact_multi_plane.to_csv(OUT_DIR / "exact_duplicates_multiple_planes.csv", index=False)

    exact_cross = []
    for _, row in exact.iterrows():
        splits = set(str(row.get("splits", "")).split(";"))
        if not {"train", "test"}.issubset(splits):
            continue
        group_paths = [p.strip() for p in str(row["paths"]).split("||")]
        train_paths = [norm_path(p) for p in group_paths if "/train/" in p.replace("\\", "/")]
        test_paths = [norm_path(p) for p in group_paths if "/test/" in p.replace("\\", "/")]
        for p in train_paths:
            exact_cross.append({
                "group_id": row["group_id"],
                "class": row.get("classes", ""),
                "planes_in_group": row.get("planes", ""),
                "train_path": p,
                "test_paths_in_exact_group": " || ".join(test_paths),
                "original_internal_split": membership.get(p, "NOT_FOUND_IN_SPLIT"),
            })
    exact_detail = pd.DataFrame(exact_cross)

    pairs = pd.read_csv(AUDIT_DIR / "near_duplicate_pairs.csv")
    cross = pairs[pairs["cross_split"].astype(str).str.lower().isin(["true", "1"])] .copy()
    direct_train_paths = set()
    for _, row in cross.iterrows():
        if str(row["split_a"]) == "train":
            direct_train_paths.add(norm_path(row["path_a"]))
        if str(row["split_b"]) == "train":
            direct_train_paths.add(norm_path(row["path_b"]))

    def count_membership(path_set: set[str]) -> dict[str, int]:
        counts = defaultdict(int)
        for p in path_set:
            counts[membership.get(p, "NOT_FOUND_IN_SPLIT")] += 1
        return dict(counts)

    candidate_paths = set(candidates["train_path_norm"])
    direct_counts = count_membership(direct_train_paths)
    transitive_counts = count_membership(candidate_paths)
    exact_paths = set(exact_detail["train_path"]) if not exact_detail.empty else set()
    exact_counts = count_membership(exact_paths)

    # Pairs internal to test do not leak from training, but reveal repeated test cases.
    test_test = pairs[(pairs["split_a"] == "test") & (pairs["split_b"] == "test")]
    train_train = pairs[(pairs["split_a"] == "train") & (pairs["split_b"] == "train")]

    candidates.to_csv(OUT_DIR / "candidate_membership.csv", index=False)
    if not exact_detail.empty:
        exact_detail.to_csv(OUT_DIR / "exact_cross_split_membership.csv", index=False)

    report = []
    report.append("=== DIVISÃO INTERNA RECONSTRUÍDA ===")
    report.append(f"ImageFolder train original: {len(ds)}")
    report.append(f"Treino efetivo para atualizar pesos: {len(train_idx)}")
    report.append(f"Validação: {len(val_idx)}")
    report.append(f"Parâmetros: test_size={VAL_FRACTION}, random_state={SEED}, stratify=labels")
    report.append("")
    report.append("=== CANDIDATOS pHash DIRETAMENTE LIGADOS AO TESTE OFICIAL ===")
    report.append(f"Imagens únicas de treino diretamente ligadas: {len(direct_train_paths)}")
    report.append(f"Distribuição na divisão interna: {direct_counts}")
    report.append("")
    report.append("=== CANDIDATOS EM COMPONENTES TRANSITIVOS ===")
    report.append(f"Imagens únicas de treino candidatas: {len(candidate_paths)}")
    report.append(f"Distribuição na divisão interna: {transitive_counts}")
    report.append("")
    report.append("=== GRUPOS EXATOS COM MAIS DE UM PLANO INDICADO ===")
    report.append(f"Grupos com planos diferentes nos nomes: {len(exact_multi_plane)}")
    if len(exact_multi_plane):
        report.append(exact_multi_plane[["group_id", "n_files", "splits", "classes", "planes"]].to_string(index=False))
    report.append("")
    report.append("=== DUPLICATAS EXATAS ENTRE TRAIN E TEST ===")
    report.append(f"Arquivos de treino envolvidos: {len(exact_paths)}")
    report.append(f"Distribuição na divisão interna: {exact_counts}")
    if not exact_detail.empty:
        report.append("Detalhamento dos grupos:")
        for _, row in exact_detail.sort_values("group_id").iterrows():
            report.append(
                f"  Grupo {row['group_id']} | classe={row['class']} | planos={row['planes_in_group']}"
                f" | lado train={row['original_internal_split']}"
            )
            report.append(f"    train: {row['train_path']}")
            report.append(f"    test:  {row['test_paths_in_exact_group']}")
    report.append("")
    report.append("=== PARES INTERNOS A CADA PARTIÇÃO (pHash <= 6) ===")
    report.append(f"Pares train-train: {len(train_train)}")
    report.append(f"Pares test-test: {len(test_test)}")
    if len(test_test):
        test_paths = set(test_test["path_a"].map(norm_path)) | set(test_test["path_b"].map(norm_path))
        report.append(f"Imagens únicas de teste envolvidas em pares test-test: {len(test_paths)}")
    report.append("")
    report.append("=== CONTAGEM DE CANDIDATOS TRANSITIVOS POR CLASSE E DIVISÃO ===")
    if not candidates.empty:
        report.append(
            candidates.groupby(["class", "original_internal_split"]).size().to_string()
        )
    report.append("")
    report.append("Nota: 'candidato' pHash não equivale a duplicata confirmada ou identidade de paciente.")
    report.append("Nenhum arquivo do dataset, checkpoint ou resultado anterior foi modificado.")

    text = "\n".join(report)
    (OUT_DIR / "leakage_membership_report.txt").write_text(text, encoding="utf-8")
    print(text)
    print("\n=== ARQUIVOS GERADOS ===")
    print(OUT_DIR / "candidate_membership.csv")
    print(OUT_DIR / "exact_duplicates_multiple_planes.csv")
    if not exact_detail.empty:
        print(OUT_DIR / "exact_cross_split_membership.csv")
    print(OUT_DIR / "leakage_membership_report.txt")


if __name__ == "__main__":
    main()
