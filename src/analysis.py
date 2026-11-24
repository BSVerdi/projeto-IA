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
    colors = {"CNN": "#ff5555", "ResNet50": "#5555ff"}
    fig, ax = plt.subplots(figsize=(8, 8))
    data = [df[df.model == m].f1_macro.values for m in names]
    bp = ax.boxplot(
        data, widths=0.5, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black"), capprops=dict(color="black"),
        boxprops=dict(color="black"),
        flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="black", markersize=4),
    )
    for patch, name in zip(bp["boxes"], names):
        patch.set_facecolor(colors[name])
    ax.set_xticks([1, 2])
    ax.set_xticklabels(names)
    rng = np.random.default_rng(0)
    for i, d in enumerate(data, 1):
        ax.scatter(rng.normal(i, 0.04, len(d)), d, s=8, c="#333333", alpha=0.7, zorder=3)
    ax.set_xlabel("Modelo")
    ax.set_ylabel("macro_f1")
    ax.set_title(f"Distribuição de macro_f1 ao longo de {len(data[0])} seeds")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    fig.savefig(out_dir / "boxplot_f1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Curvas de treino e ROC da seed de referência
    fig_roc, ax_roc = plt.subplots(figsize=(8, 8))
    for name in names:
        hist_path = out_dir / "histories" / f"{name}_seed{ref_seed}.json"
        prob_path = out_dir / "probs" / f"{name}_seed{ref_seed}.npz"
        if not hist_path.exists():
            continue
        h = pd.DataFrame(json.load(open(hist_path)))
        fig, axs = plt.subplots(2, 1, figsize=(9, 14))
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

    make_confusion_matrices(out_dir, ref_seed)


def make_confusion_matrices(out_dir: Path, ref_seed: int, threshold: float = 0.5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    names = ["CNN", "ResNet50"]
    labels = ["NORMAL", "PNEUMONIA"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.6), constrained_layout=True)
    im = None
    for k, (ax, name) in enumerate(zip(axs, names)):
        path = out_dir / "probs" / f"{name}_seed{ref_seed}.npz"
        if not path.exists():
            ax.axis("off")
            continue
        z = np.load(path)
        preds = (z["p"] >= threshold).astype(int)
        cm = confusion_matrix(z["y"], preds, labels=[0, 1])
        pct = cm / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1%})", ha="center", va="center",
                        fontsize=12, color="white" if pct[i, j] > 0.5 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticks([0, 1])
        if k == 0:  # rótulos do eixo Y só no primeiro painel (evita sobreposição)
            ax.set_yticklabels(labels)
            ax.set_ylabel("Classe real")
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Classe predita")
        ax.set_title(f"{name} (seed={ref_seed})")

    if im is not None:
        fig.colorbar(im, ax=axs, shrink=0.8, label="Proporção da classe real")
    fig.savefig(out_dir / "matriz_confusao.png", dpi=150, bbox_inches="tight")
    plt.close(fig)