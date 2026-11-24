import argparse
import hashlib
import shutil
from pathlib import Path

CLASSES = ["NORMAL", "PNEUMONIA"]
SPLITS = ["train", "val", "test"]
EXTS = {".jpeg", ".jpg", ".png"}


def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unify(root: Path, dst: Path, move: bool = False):
    for s in SPLITS:
        if not (root / s).is_dir():
            raise FileNotFoundError(
                f"Não achei {root / s}. Aponte --root para a pasta que contém train/val/test.")
    op = shutil.move if move else shutil.copy2

    for cls in CLASSES:
        (dst / cls).mkdir(parents=True, exist_ok=True)

    counts = {c: 0 for c in CLASSES}
    duplicates, renamed = 0, 0
    for split in SPLITS:
        for cls in CLASSES:
            for img in sorted((root / split / cls).iterdir()):
                if not img.is_file() or img.suffix.lower() not in EXTS or img.name.startswith("._"):
                    continue

                target = dst / cls / img.name
                if target.exists():
                    if file_hash(target) == file_hash(img):
                        duplicates += 1      # mesma imagem repetida em outra pasta: ignora
                        continue
                    # mesmo nome, conteúdo diferente: preserva as duas
                    target = dst / cls / f"{img.stem}__{split}{img.suffix}"
                    renamed += 1

                op(str(img), str(target))
                counts[cls] += 1

    print("Concluído:", counts, "| total:", sum(counts.values()))
    print(f"Duplicatas idênticas ignoradas: {duplicates} | arquivos renomeados por colisão: {renamed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="UMA pasta que contenha train/val/test")
    ap.add_argument("--dst", default="data/chest_xray_unificado")
    ap.add_argument("--move", action="store_true")
    args = ap.parse_args()
    unify(Path(args.root), Path(args.dst), args.move)