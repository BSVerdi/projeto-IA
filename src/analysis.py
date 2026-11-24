import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, shapiro
from sklearn.metrics import roc_auc_score, roc_curve

METRICS = ["acc", "loss", "f1_macro", "f1_normal", "f1_pneumonia", "auc"]


def analyze(df: pd.DataFrame, out_dir: Path):
    rows = []

    for m in METRICS:
        a = df[df.model == "CNN"][m].values
        b = df[df.model == "ResNet50"][m].values
        rows.append(dict(
            metric=m,
            cnn_mean=a.mean(), cnn_std=a.std(ddof=1), cnn_median=np.median(a),
            resnet_mean=b.mean(), resnet_std=b.std(ddof=1), resnet_median=np.median(b),
            shapiro_p_cnn=shapiro(a).pvalue, shapiro_p_resnet=shapiro(b).pvalue,
            mannwhitney_p=mannwhitneyu(a, b, alternative="two-sided").pvalue))

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "stats_summary.csv", index=False)

    print("\n=== Estatísticas ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.5f}"))

    return res


def make_plots(df: pd.DataFrame, out_dir: Path, ref_seed: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ["CNN", "ResNet50"]

    # Boxplot do F1 macro
    fig, ax = plt.subplots(figsize=(5, 5))
    data = [df[df.model == m].f1_macro.values for m in names]
    ax.boxplot(data)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(names)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, 1):
        ax.scatter(rng.normal(i, 0.04, len(d)), d, s=10, c="k", alpha=0.6)
    ax.set_ylabel("macro_f1")
    ax.set_title(f"Distribuição de macro_f1 ao longo de {len(data[0])} seeds")
    fig.savefig(out_dir / "boxplot_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Curvas de treino e ROC da seed de referência
    fig_roc, ax_roc = plt.subplots(figsize=(6, 6))
    for name in names:
        hist_path = out_dir / "histories" / f"{name}_seed{ref_seed}.json"
        prob_path = out_dir / "probs" / f"{name}_seed{ref_seed}.npz"
        if not hist_path.exists():
            continue
        h = pd.DataFrame(json.load(open(hist_path)))
        fig, axs = plt.subplots(2, 1, figsize=(6, 7))
        axs[0].plot(h.epoch, h.train_loss, label="Loss Treino")
        axs[0].plot(h.epoch, h.val_loss, label="Loss Validação")
        axs[0].set_title("Loss Treino X Validação")
        axs[1].plot(h.epoch, h.train_acc, label="Acurácia Treino")
        axs[1].plot(h.epoch, h.val_acc, label="Acurácia Validação")
        axs[1].set_title("Acurácia Treino X Validação")
        axs[1].set_xlabel("Épocas")
        for a in axs:
            a.legend()
            a.grid(alpha=0.3)
        fig.savefig(out_dir / f"curvas_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

        z = np.load(prob_path)
        fpr, tpr, _ = roc_curve(z["y"], z["p"])
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(z['y'], z['p']):.3f})")

    ax_roc.plot([0, 1], [0, 1], "k--", label="Classificador aleatório")
    ax_roc.set_xlabel("Taxa de Falso Positivo")
    ax_roc.set_ylabel("Taxa de Verdadeiro Positivo")
    ax_roc.set_title("Curva ROC")
    ax_roc.legend()
    ax_roc.grid(alpha=0.3)
    fig_roc.savefig(out_dir / "roc.png", dpi=150, bbox_inches="tight")
    plt.close(fig_roc)