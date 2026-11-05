from pathlib import Path
from collections import Counter, defaultdict
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt

# Execute a partir da pasta raiz do projeto tumor_cnn.
DATA_ROOT = Path("data/brisc2025/classification_task")
SPLITS = ("train", "test")
CLASSES = ("glioma", "meningioma", "no_tumor", "pituitary")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SAMPLES_PER_CLASS = 3


def main():
    if not DATA_ROOT.is_dir():
        raise SystemExit(
            f"Dataset não encontrado em: {DATA_ROOT.resolve()}\n"
            "Execute este script na pasta raiz do projeto tumor_cnn."
        )

    counts = defaultdict(dict)
    dimensions = Counter()
    modes = Counter()
    corrupted = []
    samples = defaultdict(list)
    total = 0

    for split in SPLITS:
        for class_name in CLASSES:
            folder = DATA_ROOT / split / class_name
            if not folder.is_dir():
                print(f"[AVISO] Pasta ausente: {folder}")
                counts[split][class_name] = 0
                continue

            files = sorted(
                p for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
            )
            counts[split][class_name] = len(files)
            total += len(files)

            for path in files:
                try:
                    # verify() verifica a integridade estrutural do arquivo.
                    with Image.open(path) as image:
                        image.verify()

                    # Reabre para consultar dimensões/modo e guardar amostras.
                    with Image.open(path) as image:
                        dimensions[image.size] += 1
                        modes[image.mode] += 1
                        if len(samples[(split, class_name)]) < SAMPLES_PER_CLASS:
                            samples[(split, class_name)].append(path)
                except (UnidentifiedImageError, OSError, ValueError) as exc:
                    corrupted.append((path, str(exc)))

    print("\n=== QUANTIDADE DE IMAGENS POR CLASSE ===")
    for split in SPLITS:
        subtotal = sum(counts[split].values())
        print(f"\n{split.upper()} ({subtotal} imagens)")
        for class_name in CLASSES:
            print(f"  {class_name:12s}: {counts[split].get(class_name, 0):5d}")

    print(f"\nTOTAL DE IMAGENS ENCONTRADAS: {total}")
    print(f"ARQUIVOS CORROMPIDOS/ILEGÍVEIS: {len(corrupted)}")

    print("\n=== DIMENSÕES MAIS FREQUENTES ===")
    for size, quantity in dimensions.most_common(10):
        print(f"  {size[0]}x{size[1]}: {quantity}")

    print("\n=== MODOS DE COR ===")
    for mode, quantity in modes.most_common():
        print(f"  {mode}: {quantity}")

    if corrupted:
        print("\n=== PRIMEIROS ARQUIVOS COM PROBLEMA ===")
        for path, error in corrupted[:20]:
            print(f"  {path} ({error})")

    # Monta uma grade com até três exemplos de cada classe do treino.
    nrows = len(CLASSES)
    ncols = SAMPLES_PER_CLASS
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 11))

    for row, class_name in enumerate(CLASSES):
        for col in range(ncols):
            ax = axes[row, col]
            ax.axis("off")
            paths = samples[("train", class_name)]
            if col < len(paths):
                with Image.open(paths[col]) as image:
                    ax.imshow(image.convert("RGB"))
                ax.set_title(f"{class_name}\n{paths[col].name[:27]}", fontsize=8)
            else:
                ax.set_title(f"{class_name}\nSem amostra", fontsize=8)

    fig.suptitle("Exemplos do conjunto de treinamento — BRISC", fontsize=13)
    fig.tight_layout()
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "amostras_dataset.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"\nGrade de exemplos salva em: {output_path.resolve()}")

    if corrupted:
        print("\nATENÇÃO: revise os arquivos listados antes de iniciar o treinamento.")
    else:
        print("\nNenhum arquivo corrompido foi encontrado na verificação.")


if __name__ == "__main__":
    main()
