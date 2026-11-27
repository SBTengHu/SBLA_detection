#!/usr/bin/env python
# coding: utf-8

"""End-to-end example for SBLA Finder (Beta — Pass A).

Generates a synthetic Ly-alpha forest dataset, injects a masked bad-pixel region
to demonstrate pixel_mask interpolation, runs find_sblas_passA_beta(), saves
FITS outputs, and writes a diagnostic figure to outputs/.
"""

from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow running directly from either the repo root or examples/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sbla_beta_passA import find_sblas_passA_beta, interpolate_masked_pixels


def make_synthetic_forest(n_los=250, wl_start=3700.0, wl_end=4500.0, dwl=0.8, seed=42):
    """Build simple synthetic Ly-alpha forest sightlines for testing."""
    rng = np.random.default_rng(seed)
    wl_obs = np.arange(wl_start, wl_end, dwl, dtype=np.float64)
    n_pix = len(wl_obs)

    flux = np.ones((n_los, n_pix), dtype=np.float32)
    for i in range(n_los):
        n_lines = rng.integers(18, 40)
        centers = rng.uniform(wl_obs[0], wl_obs[-1], size=n_lines)
        widths = rng.uniform(0.6, 5.0, size=n_lines)
        taus = rng.exponential(scale=1.2, size=n_lines)
        tau_tot = np.zeros(n_pix, dtype=np.float64)
        for c0, w0, t0 in zip(centers, widths, taus):
            tau_tot += t0 * np.exp(-0.5 * ((wl_obs - c0) / w0) ** 2)
        flux[i] = np.exp(-tau_tot) + rng.normal(0.0, 0.05, size=n_pix)

    los_ids = np.arange(1000, 1000 + n_los, dtype=np.int64)
    obj_names = np.array([f"BETA_QSO_{i:04d}" for i in range(n_los)])
    return flux, wl_obs, los_ids, obj_names


def save_diagnostic_plot(bin_summary, out_png):
    """Plot calibrated flux threshold vs. redshift for each SBLA velocity scale."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    scales = np.unique(bin_summary["VEL_SBLA"])[::-1]
    for vel in scales:
        sub = bin_summary[bin_summary["VEL_SBLA"] == vel]
        ax.plot(sub["Z_MID"], sub["FLUX_THRESHOLD"], marker="o", label=f"{vel} km/s")

    ax.set_xlabel("Redshift Bin Midpoint ($z$)")
    ax.set_ylabel("Calibrated Flux Threshold ($F_{\\mathrm{th}}$)")
    ax.set_title("SBLA Beta (Pass A) — Per-Bin Calibrated Thresholds")
    ax.grid(alpha=0.3)
    ax.legend(title="SBLA Width", frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main():
    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)

    # 1. Create synthetic dataset
    flux, wl_obs, los_ids, obj_names = make_synthetic_forest()
    n_los, n_pix = flux.shape

    # Instrument resolution: wavelength-dependent array (or use a float, e.g., 75.0)
    inst_fwhm_kms = np.linspace(90.0, 65.0, n_pix)

    # 2. Optional pixel mask: 0 = good (unchanged), 1 = masked (re-interpolated)
    pixel_mask = np.zeros((n_los, n_pix), dtype=np.uint8)
    pixel_mask[:, 200:206] = 1
    flux_masked = flux.copy()
    flux_masked[pixel_mask == 1] = -999.0

    # Sanity check: mask == 0 pixels are strictly preserved
    flux_checked = interpolate_masked_pixels(flux_masked, wl_obs, pixel_mask)
    assert np.array_equal(flux_checked[pixel_mask == 0], flux_masked[pixel_mask == 0])

    # 3. Run Pass-A beta finder
    catalogue, bin_summary = find_sblas_passA_beta(
        flux=flux_masked,
        wl_obs=wl_obs,
        los_ids=los_ids,
        obj_names=obj_names,
        inst_fwhm_kms=inst_fwhm_kms,
        pixel_mask=pixel_mask,
        res_val_loop=[529, 340, 238, 174, 132],
        target_vf=0.004,
        dz=0.1,
        boundary_wl=5.0,
    )

    # 4. Save FITS tables and diagnostic plot
    cat_path = out_dir / "SBLA_Beta_Catalogue.fits"
    sum_path = out_dir / "SBLA_Beta_FluxLimits.fits"
    fig_path = out_dir / "SBLA_Beta_Thresholds.png"

    catalogue.write(cat_path, overwrite=True)
    bin_summary.write(sum_path, overwrite=True)
    save_diagnostic_plot(bin_summary, fig_path)

    print(f"\nSaved {len(catalogue)} detected SBLAs to: {cat_path}")
    print(f"Saved per-bin thresholds to:          {sum_path}")
    print(f"Saved diagnostic plot to:             {fig_path}")


if __name__ == "__main__":
    main()
