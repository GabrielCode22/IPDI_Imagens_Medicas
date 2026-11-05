#!/usr/bin/env python3
"""Prepara manifests de treino/validacao com triagem conservadora de leakage.

- NAO apaga, move ou copia imagens.
- Preserva integralmente o teste oficial (nao o usa para ajustar o modelo).
- Exclui do pool de treino/validacao todas as imagens de train conectadas a
  qualquer imagem do teste por arestas SHA-256 exatas ou pHash (Hamming <= 6)
  registradas pela auditoria anterior.
- Divide o pool restante por componentes (StratifiedGroupKFold), evitando que
  imagens conectadas por essas arestas caiam em treino e validacao diferentes.

Uso na raiz do projeto:
    python prepare_leakage_filtered_split.py

Saidas em results/leakage_filtered/.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "brisc2025" / "classification_task"
AUDIT_DIR = ROOT / "results" / "leakage_audit"
OUT_DIR = ROOT / "results" / "leakage_filtered"
RANDOM_SEED = 42
N_SPLITS = 5
TARGET_VAL_FRACTION = 0.20
EXPECTED_CLASSES = {"glioma", "meningioma", "no_tumor", "pituitary"}


def norm(value: str | Path) -> str:
    return str(Path(str(value)).expanduser().resolve())


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def parse_group_paths(value: str) -> list[str]:
    return [norm(x.strip()) for x in str(value).split("||") if x.strip()]


def plane_from_name(path: str) -> str:
    name = Path(path).stem.lower().split("_")
    if name and name[-2] in {"ax", "co", "sa"}:
        return {"ax": "axial", "co": "coronal", "sa": "sagittal"}[name[-2]]
    return "unknown"


def main() -> int:
    required = [
        DATA_ROOT / "train", DATA_ROOT / "test",
        AUDIT_DIR / "near_duplicate_pairs.csv",
        AUDIT_DIR / "exact_duplicate_groups.csv",
        AUDIT_DIR / "train_candidates_to_review.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Arquivos/pastas ausentes:\n" + "\n".join(missing))

    train_ds = ImageFolder(DATA_ROOT / "train")
    test_ds = ImageFolder(DATA_ROOT / "test")
    class_names = train_ds.classes
    if set(class_names) != EXPECTED_CLASSES:
        raise ValueError(f"Classes inesperadas: {class_names}")
    if test_ds.classes != class_names:
        raise ValueError("As classes do teste diferem das classes do treino.")

    # Um nó para cada imagem da partição original train e do teste oficial.
    train_paths = [norm(p) for p, _ in train_ds.samples]
    test_paths = [norm(p) for p, _ in test_ds.samples]
    all_paths = set(train_paths) | set(test_paths)
    split_of = {p: "train" for p in train_paths}
    split_of.update({p: "test" for p in test_paths})
    label_of = {norm(p): class_names[int(y)] for p, y in train_ds.samples}
    label_of.update({norm(p): class_names[int(y)] for p, y in test_ds.samples})

    uf = UnionFind(all_paths)
    unknown_edges = 0

    # Arestas de similaridade perceptual (inclui train-train e test-test).
    near = pd.read_csv(AUDIT_DIR / "near_duplicate_pairs.csv")
    required_near = {"path_a", "path_b", "split_a", "split_b", "hamming_distance"}
    if not required_near.issubset(near.columns):
        raise ValueError(f"Colunas ausentes em near_duplicate_pairs.csv: {required_near - set(near.columns)}")
    for row in near.itertuples(index=False):
        a, b = norm(row.path_a), norm(row.path_b)
        if a in all_paths and b in all_paths:
            uf.union(a, b)
        else:
            unknown_edges += 1

    # Arestas de duplicatas exatas SHA-256 (tambem conectam diferentes planos).
    exact = pd.read_csv(AUDIT_DIR / "exact_duplicate_groups.csv")
    if "paths" not in exact.columns:
        raise ValueError("Coluna 'paths' ausente em exact_duplicate_groups.csv")
    exact_group_count = 0
    for value in exact["paths"].dropna():
        group_paths = [p for p in parse_group_paths(value) if p in all_paths]
        if len(group_paths) > 1:
            exact_group_count += 1
            for p in group_paths[1:]:
                uf.union(group_paths[0], p)

    # Componentes globais do grafo combinado SHA-256 + pHash.
    components: dict[str, list[str]] = defaultdict(list)
    for p in all_paths:
        components[uf.find(p)].append(p)

    component_of = {p: uf.find(p) for p in all_paths}
    test_component_roots = {
        component_of[p] for p in test_paths
    }
    flagged_train = {p for p in train_paths if component_of[p] in test_component_roots}
    clean_paths = [p for p in train_paths if p not in flagged_train]

    # Verifica a contagem da auditoria anterior (informativa, nao substitui grafo).
    prior_candidates = pd.read_csv(AUDIT_DIR / "train_candidates_to_review.csv")
    prior_set = {norm(p) for p in prior_candidates["train_path"].dropna()}
    mismatch_candidates = len(flagged_train.symmetric_difference(prior_set))

    # Criar arrays de rotulos e grupos somente para o pool sem ligacao ao teste.
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    path_to_idx = {norm(p): int(y) for p, y in train_ds.samples}
    y = [path_to_idx[p] for p in clean_paths]
    groups = [component_of[p] for p in clean_paths]

    if len(set(y)) != len(class_names):
        raise RuntimeError("Uma ou mais classes desapareceram do pool filtrado.")
    if len(set(groups)) < N_SPLITS:
        raise RuntimeError("Poucos grupos para realizar a divisao agrupada.")

    # Gera cinco folds agrupados e escolhe o mais proximo de 80/20,
    # considerando tambem o balanceamento das quatro classes.
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    total_by_class = Counter(y)
    candidates = []
    dummy_x = [0] * len(clean_paths)
    for fold_idx, (fit_idx, val_idx) in enumerate(splitter.split(dummy_x, y, groups)):
        val_counts = Counter(y[i] for i in val_idx)
        size_error = abs(len(val_idx) / len(clean_paths) - TARGET_VAL_FRACTION)
        class_errors = [
            abs(val_counts.get(c, 0) / total_by_class[c] - TARGET_VAL_FRACTION)
            for c in range(len(class_names))
        ]
        class_error = sum(class_errors) / len(class_errors)
        missing_classes = sum(1 for c in range(len(class_names)) if val_counts.get(c, 0) == 0)
        score = size_error + class_error + missing_classes * 10
        candidates.append((score, fold_idx, fit_idx, val_idx, val_counts))

    _, chosen_fold, fit_idx, val_idx, val_counts = min(candidates, key=lambda x: x[0])
    fit_paths = [clean_paths[i] for i in fit_idx]
    val_paths = [clean_paths[i] for i in val_idx]

    fit_groups = {component_of[p] for p in fit_paths}
    val_groups = {component_of[p] for p in val_paths}
    if fit_groups & val_groups:
        raise RuntimeError("Vazamento interno: um componente apareceu em treino e validacao.")
    if any(component_of[p] in test_component_roots for p in fit_paths + val_paths):
        raise RuntimeError("Vazamento: um componente ligado ao teste permaneceu no pool limpo.")

    # Manifests absolutos, sem copiar nem mover os arquivos.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    def rows_for(paths: list[str], split_name: str) -> list[dict]:
        rows = []
        for p in paths:
            cls = label_of[p]
            rows.append({
                "path": p,
                "class": cls,
                "class_idx": class_to_idx[cls],
                "plane": plane_from_name(p),
                "component_id": component_of[p],
                "split": split_name,
            })
        return rows

    fit_rows = rows_for(fit_paths, "fit_train")
    val_rows = rows_for(val_paths, "validation")
    excluded_rows = rows_for(sorted(flagged_train), "excluded_test_linked_candidate")
    pool_rows = rows_for(clean_paths, "clean_pool")
    pd.DataFrame(fit_rows).to_csv(OUT_DIR / "train_manifest.csv", index=False)
    pd.DataFrame(val_rows).to_csv(OUT_DIR / "validation_manifest.csv", index=False)
    pd.DataFrame(excluded_rows).to_csv(OUT_DIR / "excluded_test_linked_candidates.csv", index=False)
    pd.DataFrame(pool_rows).to_csv(OUT_DIR / "clean_pool_manifest.csv", index=False)

    def dist(paths: list[str]) -> dict[str, int]:
        cnt = Counter(label_of[p] for p in paths)
        return {c: cnt.get(c, 0) for c in class_names}

    # Tamanho de componentes e componentes que incluem multiplas classes.
    clean_comp_sizes = Counter(component_of[p] for p in clean_paths)
    clean_component_class_sets = defaultdict(set)
    for p in clean_paths:
        clean_component_class_sets[component_of[p]].add(label_of[p])
    mixed_class_groups = sum(1 for s in clean_component_class_sets.values() if len(s) > 1)

    # Estatisticas de pares teste-teste pHash: teste permanece intocado.
    test_test_pairs = near[
        (near["split_a"].astype(str) == "test") &
        (near["split_b"].astype(str) == "test")
    ]
    test_test_images = set(test_test_pairs["path_a"].map(norm)) | set(test_test_pairs["path_b"].map(norm))

    report = {
        "method": "remove train files in combined exact-SHA256/pHash connected components touching official test; split remaining train by StratifiedGroupKFold",
        "phash_hamming_threshold": 6,
        "phash_scope": "same class and same anatomical plane, as in the prior audit",
        "random_seed": RANDOM_SEED,
        "n_splits": N_SPLITS,
        "selected_fold": int(chosen_fold),
        "original_train_images": len(train_paths),
        "official_test_images_untouched": len(test_paths),
        "removed_train_candidates_linked_to_test": len(flagged_train),
        "clean_pool_images": len(clean_paths),
        "fit_train_images": len(fit_paths),
        "validation_images": len(val_paths),
        "original_candidates_from_prior_report": len(prior_set),
        "candidate_set_symmetric_difference_vs_prior_audit": mismatch_candidates,
        "sha_exact_groups_processed": exact_group_count,
        "phash_pairs_processed": len(near),
        "unknown_edge_paths": unknown_edges,
        "fit_train_class_counts": dist(fit_paths),
        "validation_class_counts": dist(val_paths),
        "clean_pool_class_counts": dist(clean_paths),
        "fit_train_fraction_of_clean_pool": len(fit_paths) / len(clean_paths),
        "validation_fraction_of_clean_pool": len(val_paths) / len(clean_paths),
        "clean_components": len(clean_comp_sizes),
        "largest_clean_component_size": max(clean_comp_sizes.values()) if clean_comp_sizes else 0,
        "clean_components_with_multiple_classes": mixed_class_groups,
        "train_val_component_overlap": len(fit_groups & val_groups),
        "test_test_phash_pairs_preserved": len(test_test_pairs),
        "test_images_in_test_test_phash_pairs": len(test_test_images),
        "caveats": [
            "pHash candidates are visual-similarity flags, not confirmed duplicates or patient identities.",
            "All train images connected transitively to any official test image were conservatively excluded from the new fit/validation pool.",
            "The official test set was not modified; it still contains pHash-related images within test.",
            "No patient/exam identifiers exist, so patient-level independence cannot be guaranteed.",
            "The new metrics must be reported as an exploratory filtered-split comparison, not a fully patient-independent clinical estimate.",
        ],
    }
    (OUT_DIR / "grouped_split_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "=== DIVISAO FILTRADA E AGRUPADA — BRISC ===",
        "Nenhum arquivo original foi movido, apagado ou copiado.",
        f"Train original: {len(train_paths)} | teste oficial preservado: {len(test_paths)}",
        f"Candidatos de train ligados ao teste removidos do pool: {len(flagged_train)}",
        f"Pool limpo: {len(clean_paths)}",
        f"Treino efetivo: {len(fit_paths)} | validacao: {len(val_paths)}",
        f"Fold selecionado: {chosen_fold} de {N_SPLITS}",
        f"Diferenca em relacao aos 1191 candidatos do relatorio anterior: {mismatch_candidates}",
        f"Componentes no pool limpo: {len(clean_comp_sizes)}",
        f"Maior componente: {max(clean_comp_sizes.values()) if clean_comp_sizes else 0}",
        f"Componentes com mais de uma classe: {mixed_class_groups}",
        f"Sobreposicao de componentes treino-validacao: {len(fit_groups & val_groups)}",
        "",
        "Distribuicao por classe:",
        f"  Pool limpo: {dist(clean_paths)}",
        f"  Treino:     {dist(fit_paths)}",
        f"  Validacao:  {dist(val_paths)}",
        "",
        f"Pares pHash dentro do teste preservados: {len(test_test_pairs)}",
        f"Imagens do teste envolvidas nesses pares: {len(test_test_images)}",
        "",
        "ATENCAO: pHash sinaliza similaridade visual, nao identidade de paciente.",
        "O conjunto de teste permanece intocado e pode conter imagens relacionadas entre si.",
    ]
    (OUT_DIR / "grouped_split_report.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print("\n=== ARQUIVOS GERADOS ===")
    for filename in [
        "train_manifest.csv", "validation_manifest.csv",
        "excluded_test_linked_candidates.csv", "clean_pool_manifest.csv",
        "grouped_split_summary.json", "grouped_split_report.txt",
    ]:
        print(OUT_DIR / filename)
    print("\nProximo passo: conferir estes manifests antes de iniciar o novo treinamento.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
