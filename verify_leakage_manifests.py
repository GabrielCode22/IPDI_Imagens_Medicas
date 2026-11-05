#!/usr/bin/env python3
"""Valida os manifests da divisão filtrada do BRISC antes do retreinamento.

Executar na raiz do projeto:
    python verify_leakage_manifests.py

O script é somente leitura: não altera imagens, manifests ou checkpoints.
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "brisc2025" / "classification_task"
OUT = ROOT / "results" / "leakage_filtered"

FILES = {
    "train": OUT / "train_manifest.csv",
    "validation": OUT / "validation_manifest.csv",
    "pool": OUT / "clean_pool_manifest.csv",
    "excluded": OUT / "excluded_test_linked_candidates.csv",
}
REQUIRED = {"path", "class", "class_idx", "plane", "component_id", "split"}
EXPECTED_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


def fail(message: str) -> None:
    raise RuntimeError(message)


def norm_paths(series: pd.Series) -> list[str]:
    return [str(Path(str(p)).expanduser().resolve()) for p in series]


def main() -> int:
    print("=== VERIFICAÇÃO DOS MANIFESTS — BRISC FILTRADO ===")
    missing = [str(p) for p in [*FILES.values(), DATA_ROOT / "train", DATA_ROOT / "test"] if not p.exists()]
    if missing:
        fail("Arquivos/pastas ausentes:\n" + "\n".join(missing))

    frames: dict[str, pd.DataFrame] = {}
    for name, path in FILES.items():
        df = pd.read_csv(path)
        absent = REQUIRED - set(df.columns)
        if absent:
            fail(f"{path.name}: colunas ausentes: {sorted(absent)}")
        df = df.copy()
        df["path_norm"] = norm_paths(df["path"])
        df["component_id"] = df["component_id"].astype(str)
        frames[name] = df

    tr, va, pool, excluded = (frames[k] for k in ("train", "validation", "pool", "excluded"))
    original_train = ImageFolder(DATA_ROOT / "train")
    official_test = ImageFolder(DATA_ROOT / "test")
    original_paths = {str(Path(p).resolve()) for p, _ in original_train.samples}
    test_paths = {str(Path(p).resolve()) for p, _ in official_test.samples}
    class_to_idx = original_train.class_to_idx

    print(f"Classes ImageFolder: {class_to_idx}")
    print(f"Train manifest: {len(tr)}")
    print(f"Validation manifest: {len(va)}")
    print(f"Pool limpo: {len(pool)}")
    print(f"Excluídos: {len(excluded)}")
    print(f"Teste oficial (arquivos): {len(test_paths)}")

    # Unique path checks.
    for name, df in frames.items():
        if df["path_norm"].duplicated().any():
            fail(f"Há caminhos repetidos dentro de {name}_manifest.")

    tr_set, va_set = set(tr.path_norm), set(va.path_norm)
    pool_set, excluded_set = set(pool.path_norm), set(excluded.path_norm)
    if tr_set & va_set:
        fail("Há imagens presentes simultaneamente em treino e validação.")
    if tr_set | va_set != pool_set:
        fail("A união de treino e validação não corresponde ao pool limpo.")
    if pool_set & excluded_set:
        fail("Há imagens do pool limpo também marcadas como excluídas.")
    if pool_set | excluded_set != original_paths:
        missing_from_manifests = original_paths - (pool_set | excluded_set)
        unknown_in_manifests = (pool_set | excluded_set) - original_paths
        fail(
            "Pool limpo + excluídos não recompõem exatamente o train original. "
            f"Ausentes={len(missing_from_manifests)}, desconhecidos={len(unknown_in_manifests)}"
        )
    if (pool_set | excluded_set) & test_paths:
        fail("O manifesto de treino/validação contém caminho do teste oficial.")

    # File existence and labels/path consistency.
    for name, df in frames.items():
        absent_files = [p for p in df.path_norm if not Path(p).is_file()]
        if absent_files:
            fail(f"{name}: {len(absent_files)} arquivos não existem; exemplo: {absent_files[:3]}")
        for _, row in df.iterrows():
            if Path(row["path_norm"]).parent.name != str(row["class"]):
                fail(f"Classe não corresponde à pasta: {row['path_norm']}")
            expected_idx = class_to_idx.get(str(row["class"]))
            if expected_idx is None or int(row["class_idx"]) != expected_idx:
                fail(f"class_idx incompatível em: {row['path_norm']}")

    if set(original_train.classes) != set(EXPECTED_CLASSES):
        fail(f"Classes inesperadas no dataset: {original_train.classes}")

    train_groups = set(tr.component_id)
    val_groups = set(va.component_id)
    group_overlap = train_groups & val_groups
    if group_overlap:
        fail(f"Componentes compartilhados entre treino e validação: {len(group_overlap)}")

    # Split labels should be coherent.
    if set(tr.split.astype(str)) != {"fit_train"}:
        fail("Coluna split do manifesto de treino não contém somente 'fit_train'.")
    if set(va.split.astype(str)) != {"validation"}:
        fail("Coluna split do manifesto de validação não contém somente 'validation'.")

    print("\n=== DISTRIBUIÇÃO POR CLASSE ===")
    print("Treino:\n", tr["class"].value_counts().reindex(EXPECTED_CLASSES, fill_value=0).to_string())
    print("Validação:\n", va["class"].value_counts().reindex(EXPECTED_CLASSES, fill_value=0).to_string())

    print("\n=== VERIFICAÇÕES ===")
    checks = [
        ("Caminhos únicos em cada manifesto", True),
        ("Treino e validação sem imagens repetidas", not bool(tr_set & va_set)),
        ("Treino + validação = pool limpo", tr_set | va_set == pool_set),
        ("Pool limpo e excluídos recompõem train original", pool_set | excluded_set == original_paths),
        ("Nenhum caminho do teste nos manifests de treino", not bool((pool_set | excluded_set) & test_paths)),
        ("Todos os arquivos existem", True),
        ("Classes e índices coerentes", True),
        ("Nenhum componente compartilhado treino/validação", len(group_overlap) == 0),
        ("Rótulos da coluna split coerentes", True),
    ]
    for label, ok in checks:
        print(f"{'OK' if ok else 'FALHOU'} — {label}")

    print("\nRESUMO: manifests consistentes para iniciar o treinamento filtrado.")
    print("Aviso: pHash é triagem de similaridade visual, não identificação de pacientes.")
    print("Aviso: o teste oficial pode conter dependências internas entre suas próprias imagens.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\nERRO NA VERIFICAÇÃO: {exc}", file=sys.stderr)
        raise SystemExit(1)
