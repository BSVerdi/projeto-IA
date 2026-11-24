import argparse
import shutil
from pathlib import Path

CLASSES = ["NORMAL", "PNEUMONIA"]
SPLITS = ["train", "val", "test"]

def unify(root: Path, dst: Path, move: bool = False):
    for s in SPLITS:
        if not (root / s).is_dir():
            raise FileNotFoundError(f"Não achei {root / s}. Aponte --root para a pasta que contém train/val/test.")
    op = shutil.move if move else shutil.copy2

    for cls in CLASSES:
        (dst / cls).mkdir(parents=True, exist_ok=True)

    counts = {c: 0 for c in CLASSES}
    for split in SPLITS:
        for cls in CLASSES:
            for img in sorted((root / split / cls).iterdir()):
                if img.suffix.lower() != ".jpeg" or img.name.startswith("._"):
                    continue
                op(str(img), str(dst / cls / f"{split}_{img.name}"))
                counts[cls] += 1

    print("Concluído:", counts, "| total:", sum(counts.values()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="UMA pasta que contenha train/val/test")
    ap.add_argument("--dst", default="chest_xray_unificado")
    ap.add_argument("--move", action="store_true")
    args = ap.parse_args()
    unify(Path(args.root), Path(args.dst), args.move)