import argparse
from pathlib import Path

import pandas as pd
import torch

from src.analysis import analyze, make_plots
from src.dataset import collect_files, make_splits
from src.train import HPARAMS, run_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="pasta unificada com NORMAL/ e PNEUMONIA/")
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ref_seed", type=int, default=7, help="seed usada nas curvas/ROC")
    ap.add_argument("--models", nargs="+", default=["CNN", "ResNet50"], choices=list(HPARAMS))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    paths, labels = collect_files(Path(args.data_dir))
    print(f"Total: {len(paths)} | NORMAL={int((labels == 0).sum())} "
          f"PNEUMONIA={int((labels == 1).sum())}")
    splits = make_splits(paths, labels)
    print("Treino/Val/Teste:", [len(s[0]) for s in splits])

    # retoma execuções já concluídas
    runs_csv = out_dir / "runs.csv"
    rows = pd.read_csv(runs_csv).to_dict("records") if runs_csv.exists() else []
    done = {(r["model"], int(r["seed"])) for r in rows}

    for name in args.models:
        for seed in range(args.seeds):
            if (name, seed) in done:
                print(f"Pulando {name} seed={seed} (já feito)")
                continue
            print(f"\n>>> {name} | seed {seed}")
            rows.append(run_one(name, seed, splits, device, args.workers, out_dir))
            pd.DataFrame(rows).to_csv(runs_csv, index=False)  # checkpoint a cada execução

    df = pd.DataFrame(rows)
    if set(df.model) >= {"CNN", "ResNet50"}:
        analyze(df, out_dir)
        make_plots(df, out_dir, args.ref_seed)
    else:
        print("Estatística/gráficos exigem os dois modelos em runs.csv.")


if __name__ == "__main__":  # necessário no Windows (DataLoader com workers)
    main()