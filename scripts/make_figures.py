"""Reproduce the figures in README.md.  Usage: python scripts/make_figures.py"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from urnloop import critical_lambda, mean_field_drift, simulate  # noqa: E402

FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9cc3ef", "#2a78d6", "#0d3a73"])

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.bbox": "tight",
})


def sample_paths():
    alphas = [0.7, 1.0, 1.5]
    titles = ["α = 0.7: diversity survives", "α = 1: Pólya urn, random limit", "α = 1.5: lock-in"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, a, title in zip(axes, alphas, titles):
        out = simulate(a, 0.0, K=3, n_steps=20_000, record_every=20, seed=7)
        h = out["history"][:, 0, :]
        for i in range(3):
            ax.plot(out["times"], h[:, i], color=SERIES[i], lw=2, label=f"mode {i + 1}")
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlim(0, out["times"][-1])
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel("round n")
    axes[0].set_ylabel("share of training data")
    axes[0].set_ylim(0, 1)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Pure self-training (λ = 0), K = 3, one run each", y=1.03, color=INK)
    fig.savefig(FIG / "sample_paths.png")
    plt.close(fig)


def phase_diagram():
    K = 3
    alphas = np.linspace(0.5, 3.0, 26)
    lams = np.linspace(0.0, 0.8, 33)
    A, L = np.meshgrid(alphas, lams)
    out = simulate(A, L, K=K, n_steps=20_000, n_runs=20, seed=11)
    dom = (out["shares"].max(axis=-1) - 1 / K) / (1 - 1 / K)
    dom = dom.mean(axis=-1)

    fig, ax = plt.subplots(figsize=(6, 4.2))
    mesh = ax.pcolormesh(A, L, dom, cmap=SEQ, vmin=0, vmax=1, shading="nearest")
    a_fine = np.linspace(1.0, 3.0, 200)
    ax.plot(a_fine, critical_lambda(a_fine), color=SERIES[1], lw=2, label="λ = 1 − 1/α  (mean-field threshold)")
    ax.set_xlabel("sharpening α  (= 1 / sampling temperature)")
    ax.set_ylabel("fresh-data fraction λ")
    ax.grid(False)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label("dominance after 20,000 rounds\n(0 = uniform, 1 = one mode only)")
    ax.set_title(f"Phase diagram, K = {K}, uniform start, mean of 20 runs", fontsize=10, color=INK)
    fig.savefig(FIG / "phase_diagram.png")
    plt.close(fig)


def bifurcation():
    alpha = 3.0
    lams = np.linspace(0.0, 0.8, 161)
    xs = np.linspace(1e-6, 1 - 1e-6, 200_001)
    X = np.stack([xs, 1 - xs], -1)
    up, low = [], []
    for lam in lams:
        F = mean_field_drift(X, alpha, lam)[:, 0]
        roots = xs[:-1][np.sign(F[:-1]) != np.sign(F[1:])]
        off = roots[np.abs(roots - 0.5) > 1e-3]
        up.append(off.max() if off.size else np.nan)
        low.append(off.min() if off.size else np.nan)
    lc = float(critical_lambda(alpha))

    lam_sim = np.linspace(0.0, 0.8, 17)
    sim = simulate(alpha, lam_sim, K=2, n_steps=20_000, n_runs=30, seed=3)["shares"][..., 0]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(lams, up, color=SERIES[0], lw=2, label="stable fixed points")
    ax.plot(lams, low, color=SERIES[0], lw=2)
    ax.plot([0, lc], [0.5, 0.5], color=MUTED, lw=2, ls="--", label="uniform (unstable)")
    ax.plot([lc, 0.8], [0.5, 0.5], color=SERIES[0], lw=2)
    jitter = np.random.default_rng(0).uniform(-0.006, 0.006, sim.shape)
    ax.scatter((lam_sim[:, None] + jitter).ravel(), sim.ravel(), s=9, color=SERIES[1],
               alpha=0.6, lw=0, label="simulation, n = 20,000 (30 runs per λ)")
    ax.axvline(lc, color=MUTED, lw=0.8)
    ax.text(lc + 0.01, 0.03, "λc = 1 − 1/α", color=MUTED, fontsize=9)
    ax.set_xlabel("fresh-data fraction λ")
    ax.set_ylabel("share of mode 1")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.set_title("K = 2, α = 3: pitchfork at the threshold", fontsize=10, color=INK)
    fig.savefig(FIG / "bifurcation.png")
    plt.close(fig)


if __name__ == "__main__":
    sample_paths()
    bifurcation()
    phase_diagram()
    print("figures written to", FIG)
