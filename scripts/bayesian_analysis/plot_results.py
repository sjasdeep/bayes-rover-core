"""
plot_results.py
---------------
Generate all paper figures from the saved Bayesian analysis outputs.

Run from repo root:
    python scripts/bayesian_analysis/plot_results.py

Figures saved to outputs/bayesian_analysis/figures/:
    1. error_distributions.pdf    — raw error histograms per dim
    2. error_posterior.pdf        — posterior over mu and sigma
    3. posterior_predictive_check.pdf — PPC: simulated vs observed errors
    4. risk_heatmap.pdf           — P(fail | x0) over (py, theta) grid
    5. risk_vs_hj.pdf             — Bayesian risk vs HJ BRT (if available)
    6. risk_1d_slices.pdf         — P(fail) vs py at several theta values
"""

import sys
import json
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

warnings.filterwarnings("ignore")

REPO_ROOT  = Path(__file__).resolve().parent.parent.parent
OUTPUT_DIR = REPO_ROOT / "outputs" / "bayesian_analysis"
FIG_DIR    = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DIM_LABELS = [r"$p_x$ error (m)", r"$p_y$ error (m)", r"$\theta$ error (rad)"]
DIM_NAMES  = ["px", "py", "theta"]

plt.rcParams.update({
    "font.family":    "serif",
    "font.size":      11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "figure.dpi":     150,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
})

BLUE  = "#2166ac"
RED   = "#d6604d"
GREEN = "#4dac26"
GRAY  = "#888888"
PURPLE = "#7b3294"


# ============================================================
# Load outputs
# ============================================================

def load_all():
    needed = ["risk_grid.npy", "py_range.npy", "theta_range.npy",
              "errors.npy", "posterior_samples.pkl", "metadata.json"]
    for f in needed:
        if not (OUTPUT_DIR / f).exists():
            raise FileNotFoundError(
                f"Missing {f}. Run run_analysis.py first."
            )

    out = {
        "risk_grid":   np.load(OUTPUT_DIR / "risk_grid.npy"),
        "py_range":    np.load(OUTPUT_DIR / "py_range.npy"),
        "theta_range": np.load(OUTPUT_DIR / "theta_range.npy"),
        "errors":      np.load(OUTPUT_DIR / "errors.npy"),
    }
    with open(OUTPUT_DIR / "posterior_samples.pkl", "rb") as f:
        out["posterior_samples"] = pickle.load(f)
    with open(OUTPUT_DIR / "metadata.json") as f:
        out["meta"] = json.load(f)

    hj_path = OUTPUT_DIR / "hj_value_grid.npy"
    out["hj_grid"] = np.load(hj_path) if hj_path.exists() else None

    print("Loaded outputs. Risk grid shape:", out["risk_grid"].shape)
    return out


# ============================================================
# Figure 1: Raw error distributions
# ============================================================

def fig_error_distributions(errors):
    D = errors.shape[1]
    fig, axes = plt.subplots(1, D, figsize=(4.5*D, 4), sharey=False)
    colors = [BLUE, RED, GREEN]

    for d, ax in enumerate(axes):
        c = colors[d % len(colors)]
        ax.hist(errors[:, d], bins=60, density=True, color=c, alpha=0.75,
                edgecolor="white", linewidth=0.5)
        mean_d, std_d = errors[:, d].mean(), errors[:, d].std()
        ax.axvline(mean_d, color="k", lw=1.5, ls="--", label=f"mean={mean_d:.4f}")
        ax.axvline(mean_d - 2*std_d, color=GRAY, lw=1, ls=":", alpha=0.8)
        ax.axvline(mean_d + 2*std_d, color=GRAY, lw=1, ls=":", alpha=0.8,
                   label=f"±2σ (σ={std_d:.4f})")
        ax.set_xlabel(DIM_LABELS[d] if d < len(DIM_LABELS) else f"dim {d}")
        ax.set_ylabel("Density" if d == 0 else "")
        ax.set_title(f"Perception Error — {DIM_NAMES[d] if d < len(DIM_NAMES) else d}")
        ax.legend(fontsize=8)

    fig.suptitle("Observed Perception Error Distributions (RoVer-CoRe Simulator)", y=1.01)
    fig.tight_layout()
    _save(fig, "error_distributions")


# ============================================================
# Figure 2: Posterior distributions
# ============================================================

def fig_error_posterior(posterior_samples, errors):
    mu_arr    = np.array([s["mu"]    for s in posterior_samples])
    sigma_arr = np.array([s["sigma"] for s in posterior_samples])
    D = mu_arr.shape[1]

    fig, axes = plt.subplots(2, D, figsize=(4.5*D, 7))
    if D == 1:
        axes = axes.reshape(2, 1)
    colors = [BLUE, RED, GREEN]

    for d in range(D):
        c   = colors[d % len(colors)]
        lbl = DIM_NAMES[d] if d < len(DIM_NAMES) else f"dim{d}"

        # Row 0: posterior for mu
        ax = axes[0, d]
        ax.hist(mu_arr[:, d], bins=50, density=True, color=c, alpha=0.7, edgecolor="white")
        m  = mu_arr[:, d].mean()
        lo, hi = np.percentile(mu_arr[:, d], [2.5, 97.5])
        ax.axvline(m,  color="k", lw=2,   ls="--", label=f"mean={m:.5f}")
        ax.axvspan(lo, hi, alpha=0.2, color=c, label=f"95% CI [{lo:.4f},{hi:.4f}]")
        ax.axvline(0, color=GRAY, lw=1, ls=":", alpha=0.6)
        ax.set_title(f"{lbl}: posterior for $\\mu$")
        ax.set_xlabel(r"$\mu$")
        ax.set_ylabel("Density" if d == 0 else "")
        ax.legend(fontsize=7.5)

        # Row 1: posterior for sigma
        ax2 = axes[1, d]
        ax2.hist(sigma_arr[:, d], bins=50, density=True, color=c, alpha=0.7, edgecolor="white")
        m2 = sigma_arr[:, d].mean()
        lo2, hi2 = np.percentile(sigma_arr[:, d], [2.5, 97.5])
        ax2.axvline(m2,  color="k", lw=2,    ls="--", label=f"mean={m2:.5f}")
        ax2.axvspan(lo2, hi2, alpha=0.2, color=c, label=f"95% CI [{lo2:.4f},{hi2:.4f}]")
        # Overlay empirical std
        ax2.axvline(errors[:, d].std(), color="k", lw=1, ls=":",
                    alpha=0.8, label=f"empirical σ={errors[:,d].std():.5f}")
        ax2.set_title(f"{lbl}: posterior for $\\sigma$")
        ax2.set_xlabel(r"$\sigma$")
        ax2.set_ylabel("Density" if d == 0 else "")
        ax2.legend(fontsize=7.5)

    axes[0, 0].set_ylabel(r"Posterior $p(\mu \mid \mathbf{e})$")
    axes[1, 0].set_ylabel(r"Posterior $p(\sigma \mid \mathbf{e})$")
    fig.suptitle("Posterior Distributions for Perception Error Parameters", y=1.01)
    fig.tight_layout()
    _save(fig, "error_posterior")


# ============================================================
# Figure 3: Posterior predictive check
# ============================================================

def fig_ppc(posterior_samples, errors, n_ppc=1000):
    rng = np.random.default_rng(42)
    D   = errors.shape[1]

    # Draw errors from posterior predictive
    idx_draws = rng.integers(0, len(posterior_samples), size=n_ppc)
    ppc_errors = np.array([
        rng.normal(posterior_samples[i]["mu"], posterior_samples[i]["sigma"])
        for i in idx_draws
    ])  # (n_ppc, D)

    fig, axes = plt.subplots(1, D, figsize=(4.5*D, 4))
    colors = [BLUE, PURPLE, GREEN]

    for d, ax in enumerate(axes):
        c = colors[d % len(colors)]
        # Sub-sample observed for clarity
        obs_sub = rng.choice(errors[:, d], size=min(5000, len(errors)), replace=False)
        ax.hist(obs_sub,        bins=60, density=True, color=RED,  alpha=0.55,
                label="Observed", edgecolor="white", linewidth=0.3)
        ax.hist(ppc_errors[:, d], bins=60, density=True, color=c, alpha=0.55,
                label="Posterior predictive", edgecolor="white", linewidth=0.3)
        ax.set_xlabel(DIM_LABELS[d] if d < len(DIM_LABELS) else f"dim {d}")
        ax.set_ylabel("Density" if d == 0 else "")
        ax.set_title(f"PPC — {DIM_NAMES[d] if d < len(DIM_NAMES) else d}")
        ax.legend(fontsize=8)

    fig.suptitle("Posterior Predictive Check: Simulated vs. Observed Errors", y=1.01)
    fig.tight_layout()
    _save(fig, "posterior_predictive_check")


# ============================================================
# Figure 4: Risk heatmap
# ============================================================

def fig_risk_heatmap(risk_grid, py_range, theta_range):
    mean_grid = risk_grid[:, :, 0]
    ci_width  = risk_grid[:, :, 3] - risk_grid[:, :, 2]
    theta_deg = np.degrees(theta_range)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Panel A: posterior mean P(fail)
    ax = axes[0]
    im = ax.pcolormesh(theta_deg, py_range, mean_grid,
                       cmap="RdYlGn_r", vmin=0, vmax=1, shading="auto")
    ax.set_xlabel(r"Initial heading $\theta_0$ (deg)")
    ax.set_ylabel(r"Initial lateral position $p_{y,0}$ (m)")
    ax.set_title(r"Posterior Mean $P(\mathrm{fail} \mid x_0)$")
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.4, label="center")
    ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.4)
    plt.colorbar(im, ax=ax, label="P(failure)")
    ax.legend(fontsize=8)

    # Contour at p=0.5
    try:
        cs = ax.contour(theta_deg, py_range, mean_grid, levels=[0.5],
                        colors=["white"], linewidths=[2])
        ax.clabel(cs, fmt="p=0.5", fontsize=9, colors="white")
    except Exception:
        pass

    # Panel B: 95% CI width
    ax2 = axes[1]
    im2 = ax2.pcolormesh(theta_deg, py_range, ci_width,
                          cmap="viridis", vmin=0, vmax=0.5, shading="auto")
    ax2.set_xlabel(r"Initial heading $\theta_0$ (deg)")
    ax2.set_ylabel(r"Initial lateral position $p_{y,0}$ (m)")
    ax2.set_title("95% Credible Interval Width")
    ax2.axhline(0, color="w", lw=0.8, ls="--", alpha=0.4)
    ax2.axvline(0, color="w", lw=0.8, ls="--", alpha=0.4)
    plt.colorbar(im2, ax=ax2, label="CI width")

    fig.suptitle("Bayesian Posterior Predictive Failure Risk over Initial State Space")
    fig.tight_layout()
    _save(fig, "risk_heatmap")


# ============================================================
# Figure 5: Risk vs HJ BRT (or 1D slice if no HJ)
# ============================================================

def fig_risk_vs_hj(risk_grid, py_range, theta_range, hj_grid):
    if hj_grid is None:
        _fig_risk_1d_slices(risk_grid, py_range, theta_range)
        return

    mean_grid = risk_grid[:, :, 0]
    brt_unsafe = (hj_grid < 0).astype(float)
    theta_deg  = np.degrees(theta_range)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: BRT
    ax = axes[0]
    ax.pcolormesh(theta_deg, py_range, brt_unsafe,
                  cmap="RdYlGn_r", vmin=0, vmax=1, shading="auto")
    ax.set_title("HJ BRT: Worst-Case Unsafe (red = 1)")
    ax.set_xlabel(r"$\theta_0$ (deg)")
    ax.set_ylabel(r"$p_{y,0}$ (m)")

    # Panel 2: Bayesian
    ax2 = axes[1]
    im = ax2.pcolormesh(theta_deg, py_range, mean_grid,
                        cmap="RdYlGn_r", vmin=0, vmax=1, shading="auto")
    ax2.set_title(r"Bayesian $P(\mathrm{fail} \mid x_0)$")
    ax2.set_xlabel(r"$\theta_0$ (deg)")
    ax2.set_ylabel(r"$p_{y,0}$ (m)")
    plt.colorbar(im, ax=ax2)

    # Panel 3: Conservatism gap
    ax3 = axes[2]
    diff = mean_grid - brt_unsafe
    im3 = ax3.pcolormesh(theta_deg, py_range, diff,
                         cmap="bwr", vmin=-0.6, vmax=0.6, shading="auto")
    ax3.set_title(r"$P(\mathrm{fail})_\mathrm{Bayes} - \mathbf{1}_\mathrm{BRT}$")
    ax3.set_xlabel(r"$\theta_0$ (deg)")
    ax3.set_ylabel(r"$p_{y,0}$ (m)")
    cb = plt.colorbar(im3, ax=ax3)
    cb.set_label("Bayesian – HJ (blue = BRT conservative)")

    # Annotation
    n_brt_unsafe  = int(brt_unsafe.sum())
    n_bayes_high  = int((mean_grid > 0.5).sum())
    fig.text(0.5, -0.02,
             f"BRT unsafe: {n_brt_unsafe}/{mean_grid.size} states  |  "
             f"Bayesian P(fail)>0.5: {n_bayes_high}/{mean_grid.size} states",
             ha="center", fontsize=10, color=GRAY)

    fig.suptitle("Bayesian Risk vs. RoVer-CoRe HJ Worst-Case BRT")
    fig.tight_layout()
    _save(fig, "risk_vs_hj")


def _fig_risk_1d_slices(risk_grid, py_range, theta_range):
    """Fallback when no HJ value: 1D slices of P(fail) vs py at several theta."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = [BLUE, RED, GREEN, "#ff7f00", "#984ea3", "#a65628", GRAY]

    # Panel A: P(fail) vs py, one curve per theta
    ax = axes[0]
    for j, theta in enumerate(theta_range):
        mean_col = risk_grid[:, j, 0]
        lo_col   = risk_grid[:, j, 2]
        hi_col   = risk_grid[:, j, 3]
        c = colors[j % len(colors)]
        ax.fill_between(py_range, lo_col, hi_col, alpha=0.15, color=c)
        ax.plot(py_range, mean_col, color=c, lw=2,
                label=f"θ={np.degrees(theta):+.0f}°")
    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5, label="p=0.5")
    ax.set_xlabel(r"Initial lateral position $p_{y,0}$ (m)")
    ax.set_ylabel(r"$P(\mathrm{fail} \mid x_0)$")
    ax.set_title(r"P(fail) vs $p_{y,0}$, sliced by $\theta_0$")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8, ncol=2)

    # Panel B: P(fail) vs theta at py=0 (center) and py=±0.5
    ax2 = axes[1]
    target_pys = [0.0, 0.5, -0.5, 1.0, -1.0]
    theta_deg  = np.degrees(theta_range)
    for k, py_target in enumerate(target_pys):
        i = int(np.argmin(np.abs(py_range - py_target)))
        c = colors[k % len(colors)]
        ax2.fill_between(theta_deg, risk_grid[i, :, 2], risk_grid[i, :, 3],
                         alpha=0.15, color=c)
        ax2.plot(theta_deg, risk_grid[i, :, 0], color=c, lw=2,
                 label=f"py={py_range[i]:+.2f}m")
    ax2.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5)
    ax2.set_xlabel(r"Initial heading $\theta_0$ (deg)")
    ax2.set_ylabel(r"$P(\mathrm{fail} \mid x_0)$")
    ax2.set_title(r"P(fail) vs $\theta_0$, sliced by $p_{y,0}$")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=8)

    fig.suptitle("Posterior Predictive Failure Risk — 1D Slices through State Space")
    fig.tight_layout()
    _save(fig, "risk_1d_slices")


# ============================================================
# Helper
# ============================================================

def _save(fig, name: str):
    for ext in ["pdf", "png"]:
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
    print(f"  Saved: {FIG_DIR}/{name}.pdf")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    print("="*55)
    print("  Generating paper figures")
    print("="*55)

    r = load_all()

    print("\n--- Figure 1: Raw error distributions ---")
    fig_error_distributions(r["errors"])

    print("\n--- Figure 2: Error posterior ---")
    fig_error_posterior(r["posterior_samples"], r["errors"])

    print("\n--- Figure 3: Posterior predictive check ---")
    fig_ppc(r["posterior_samples"], r["errors"])

    print("\n--- Figure 4: Risk heatmap ---")
    fig_risk_heatmap(r["risk_grid"], r["py_range"], r["theta_range"])

    print("\n--- Figure 5: Risk vs HJ / 1D slices ---")
    fig_risk_vs_hj(r["risk_grid"], r["py_range"], r["theta_range"], r["hj_grid"])

    print(f"\nAll figures saved to {FIG_DIR}/")
    print("Files:")
    for f in sorted(FIG_DIR.glob("*.pdf")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
