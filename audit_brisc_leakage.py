#!/usr/bin/env python3
"""Audit exact and perceptual near-duplicates across BRISC classification splits.

Read-only: this script never deletes, moves, or modifies dataset files or model weights.
It mirrors a published BRISC screen for near-duplicates: 64-bit pHash, grouped by
class and MRI plane, with Hamming distance <= 6. A pHash match is only a visual
similarity flag, NOT proof that two images belong to the same patient.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, ImageDraw

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PLANE_RE = re.compile(r"_(ax|co|sa)_t1(?:\.|_|$)", re.IGNORECASE)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def plane_from_name(name: str) -> str:
    m = PLANE_RE.search(name)
    if not m:
        # Fallback for filenames that still contain a separated plane token.
        m = re.search(r"(?:^|[_-])(ax|co|sa)(?:[_-]|$)", name, re.IGNORECASE)
    if not m:
        return "unknown"
    return {"ax": "axial", "co": "coronal", "sa": "sagittal"}[m.group(1).lower()]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Auditoria somente leitura de duplicatas do BRISC.")
    parser.add_argument("--root", default="data/brisc2025/classification_task",
                        help="Pasta classification_task (padrão: data/brisc2025/classification_task)")
    parser.add_argument("--out", default="results/leakage_audit",
                        help="Pasta de saída (padrão: results/leakage_audit)")
    parser.add_argument("--hamming", type=int, default=6,
                        help="Distância máxima pHash (padrão: 6; triagem exploratória)")
    parser.add_argument("--max-contact-pairs", type=int, default=36,
                        help="Máximo de pares exibidos na folha de contato (padrão: 36)")
    args = parser.parse_args()

    try:
        import imagehash
    except ImportError:
        print("Dependência ausente. Instale com: python -m pip install ImageHash")
        return 2

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if not (root / "train").is_dir() or not (root / "test").is_dir():
        print(f"ERRO: não encontrei as pastas train/ e test/ em: {root}")
        return 2
    if args.hamming < 0 or args.hamming > 64:
        print("ERRO: --hamming deve estar entre 0 e 64.")
        return 2

    records: list[dict[str, Any]] = []
    unreadable: list[dict[str, str]] = []
    for split in ("train", "test"):
        split_dir = root / split
        for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                    continue
                rec = {
                    "index": len(records),
                    "split": split,
                    "class": class_dir.name,
                    "plane": plane_from_name(path.name),
                    "path": str(path.resolve()),
                    "filename": path.name,
                }
                try:
                    rec["sha256"] = sha256_file(path)
                    with Image.open(path) as im:
                        im.verify()
                    with Image.open(path) as im:
                        # Match the published procedure: grayscale pHash, 64 bits.
                        rec["phash"] = int(str(imagehash.phash(im.convert("L"), hash_size=8)), 16)
                except Exception as e:
                    unreadable.append({"path": str(path.resolve()), "error": str(e)})
                    continue
                records.append(rec)

    if not records:
        print("Nenhuma imagem legível foi encontrada.")
        return 2

    out.mkdir(parents=True, exist_ok=True)
    print("\n=== AUDITORIA DE DUPLICATAS BRISC — SOMENTE LEITURA ===")
    print(f"Raiz: {root}")
    print(f"Imagens legíveis: {len(records)} | não legíveis: {len(unreadable)}")
    print(f"pHash: 64 bits | agrupamento por classe + plano | Hamming <= {args.hamming}")

    # Exact byte duplicates using SHA-256, regardless of class/plane.
    sha_groups: dict[str, list[int]] = defaultdict(list)
    for r in records:
        sha_groups[r["sha256"]].append(r["index"])
    exact_groups = [ids for ids in sha_groups.values() if len(ids) > 1]

    exact_rows: list[dict[str, Any]] = []
    uf = UnionFind(len(records))
    for gid, ids in enumerate(exact_groups, start=1):
        for i in ids[1:]:
            uf.union(ids[0], i)
        exact_rows.append({
            "group_id": gid,
            "n_files": len(ids),
            "splits": ";".join(sorted({records[i]["split"] for i in ids})),
            "classes": ";".join(sorted({records[i]["class"] for i in ids})),
            "planes": ";".join(sorted({records[i]["plane"] for i in ids})),
            "paths": " || ".join(records[i]["path"] for i in ids),
        })

    exact_cross_pairs = 0
    exact_train_linked_to_test: set[int] = set()
    for ids in exact_groups:
        trains = [i for i in ids if records[i]["split"] == "train"]
        tests = [i for i in ids if records[i]["split"] == "test"]
        if trains and tests:
            exact_cross_pairs += len(trains) * len(tests)
            exact_train_linked_to_test.update(trains)

    # Perceptual near-duplicate search, using same class and plane as the paper.
    strata: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r in records:
        if r["plane"] != "unknown":
            strata[(r["class"], r["plane"])].append(r["index"])

    near_rows: list[dict[str, Any]] = []
    near_cross_pairs = 0
    near_train_linked_to_test: set[int] = set()
    comparisons = 0
    for (cls, plane), ids in sorted(strata.items()):
        for a, b in itertools.combinations(ids, 2):
            comparisons += 1
            # Exact duplicates have already been identified by SHA-256.
            d = (records[a]["phash"] ^ records[b]["phash"]).bit_count()
            if d <= args.hamming:
                uf.union(a, b)
                cross = records[a]["split"] != records[b]["split"]
                if cross:
                    near_cross_pairs += 1
                    train_i = a if records[a]["split"] == "train" else b
                    near_train_linked_to_test.add(train_i)
                near_rows.append({
                    "class": cls,
                    "plane": plane,
                    "hamming_distance": d,
                    "cross_split": cross,
                    "path_a": records[a]["path"],
                    "split_a": records[a]["split"],
                    "path_b": records[b]["path"],
                    "split_b": records[b]["split"],
                })

    # Transitive groups: any train item in a component containing a test item is flagged.
    components: dict[int, list[int]] = defaultdict(list)
    for r in records:
        components[uf.find(r["index"])].append(r["index"])
    cross_components = [ids for ids in components.values()
                        if any(records[i]["split"] == "train" for i in ids)
                        and any(records[i]["split"] == "test" for i in ids)]
    transitive_train_candidates: set[int] = set()
    candidate_rows: list[dict[str, Any]] = []
    for ids in cross_components:
        trains = [i for i in ids if records[i]["split"] == "train"]
        tests = [i for i in ids if records[i]["split"] == "test"]
        transitive_train_candidates.update(trains)
        for ti in trains:
            candidate_rows.append({
                "train_path": records[ti]["path"],
                "class": records[ti]["class"],
                "plane": records[ti]["plane"],
                "test_matches_in_component": len(tests),
                "test_paths_in_component": " || ".join(records[x]["path"] for x in tests),
                "note": "CANDIDATO_PARA_REVISAO; nao remover automaticamente",
            })

    exact_columns = ["group_id", "n_files", "splits", "classes", "planes", "paths"]
    near_columns = ["class", "plane", "hamming_distance", "cross_split", "path_a", "split_a", "path_b", "split_b"]
    candidate_columns = ["train_path", "class", "plane", "test_matches_in_component", "test_paths_in_component", "note"]
    write_csv(out / "exact_duplicate_groups.csv", exact_rows, exact_columns)
    write_csv(out / "near_duplicate_pairs.csv", near_rows, near_columns)
    write_csv(out / "train_candidates_to_review.csv", candidate_rows, candidate_columns)
    write_csv(out / "unreadable_images.csv", unreadable, ["path", "error"])

    # A contact sheet helps manually inspect the closest cross-split pairs.
    cross_near = [r for r in near_rows if r["cross_split"]]
    cross_near.sort(key=lambda x: (x["hamming_distance"], x["class"], x["plane"]))
    selected = cross_near[:max(0, args.max_contact_pairs)]
    if selected:
        thumb_w, thumb_h, label_h = 220, 220, 58
        cols = 2
        rows_n = len(selected)
        sheet = Image.new("RGB", (cols * thumb_w, rows_n * (thumb_h + label_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, pair in enumerate(selected):
            for side, key in enumerate(("path_a", "path_b")):
                p = Path(pair[key])
                try:
                    with Image.open(p) as im:
                        thumb = ImageOps.contain(im.convert("RGB"), (thumb_w - 8, thumb_h - 8))
                    x = side * thumb_w + (thumb_w - thumb.width) // 2
                    y = j * (thumb_h + label_h) + (thumb_h - thumb.height) // 2
                    sheet.paste(thumb, (x, y))
                    caption = f"{pair['split_a'] if side == 0 else pair['split_b']}: {p.name[:28]}"
                    draw.text((side * thumb_w + 4, j * (thumb_h + label_h) + thumb_h + 3), caption, fill="black")
                except Exception:
                    pass
            draw.text((4, j * (thumb_h + label_h) + thumb_h + 32),
                      f"{pair['class']} | {pair['plane']} | Hamming={pair['hamming_distance']}", fill="darkred")
        sheet.save(out / "near_duplicate_contact_sheet.png")

    # Per-split and plane counts.
    split_counts = defaultdict(int)
    plane_unknown = defaultdict(int)
    for r in records:
        split_counts[r["split"]] += 1
        if r["plane"] == "unknown":
            plane_unknown[r["split"]] += 1

    summary = {
        "dataset_root": str(root),
        "read_only": True,
        "n_images_readable": len(records),
        "n_unreadable": len(unreadable),
        "split_counts": dict(split_counts),
        "unknown_plane_counts": dict(plane_unknown),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_files_in_groups": sum(len(g) for g in exact_groups),
        "exact_cross_split_groups": sum(
            1 for g in exact_groups
            if any(records[i]["split"] == "train" for i in g)
            and any(records[i]["split"] == "test" for i in g)
        ),
        "exact_train_test_pair_count": exact_cross_pairs,
        "exact_train_images_linked_to_test": len(exact_train_linked_to_test),
        "phash_hamming_threshold": args.hamming,
        "phash_comparisons_same_class_plane": comparisons,
        "phash_near_duplicate_pairs": len(near_rows),
        "phash_cross_split_pairs": near_cross_pairs,
        "phash_train_images_directly_linked_to_test": len(near_train_linked_to_test),
        "cross_split_connected_components_transitive": len(cross_components),
        "train_images_in_cross_split_components_candidates": len(transitive_train_candidates),
        "candidate_train_images_are_not_deleted": True,
        "caveats": [
            "pHash indicates visual similarity, not patient identity.",
            "The same-class/same-plane pHash screen is a heuristic; inspect flagged pairs visually.",
            "No patient/exam/series identifiers are available, so patient-level independence cannot be established.",
            "A negative duplicate screen does not prove absence of patient-level or adjacent-slice dependence.",
            "Do not remove test images or report revised performance without a prespecified protocol and rerunning evaluation.",
        ],
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== RESULTADO DA AUDITORIA ===")
    print(f"Imagens legíveis: {len(records)} (train={split_counts['train']}, test={split_counts['test']})")
    print(f"Imagens com plano não identificado: {sum(plane_unknown.values())}")
    print(f"Grupos de duplicatas exatas: {len(exact_groups)}")
    print(f"Grupos exatos que cruzam train/test: {summary['exact_cross_split_groups']}")
    print(f"Pares exatos train-test: {exact_cross_pairs}")
    print(f"Imagens de treino ligadas exatamente ao teste: {len(exact_train_linked_to_test)}")
    print(f"Comparações pHash realizadas: {comparisons}")
    print(f"Pares pHash com Hamming <= {args.hamming}: {len(near_rows)}")
    print(f"Pares pHash que cruzam train/test: {near_cross_pairs}")
    print(f"Imagens de treino diretamente ligadas ao teste por pHash: {len(near_train_linked_to_test)}")
    print(f"Imagens de treino em componentes transversais (inclui ligações transitivas): {len(transitive_train_candidates)}")
    print("\n=== ARQUIVOS GERADOS ===")
    for name in ["audit_summary.json", "exact_duplicate_groups.csv", "near_duplicate_pairs.csv",
                 "train_candidates_to_review.csv", "unreadable_images.csv"]:
        print(out / name)
    if selected:
        print(out / "near_duplicate_contact_sheet.png")
    print("\nNenhum arquivo do dataset foi apagado, movido ou alterado.")
    print("Interprete pHash como triagem de similaridade visual, não como identificação de pacientes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
