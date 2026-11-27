#!/usr/bin/env python
# coding: utf-8

"""Simplified Pass-A-only SBLA finder for beta testing or uniform mock datasets.

- Single SNR bin (no SNR calculation or cuts).
- Optional pixel mask: if provided, flux at mask == 1 is removed and linearly
  re-interpolated from mask == 0 pixels; flux at mask == 0 is left unchanged.
- Fixed target volume fraction (default 0.4%) at every velocity scale.
- Runs per redshift bin directly (Pass A style): finds the optimal flux limit
  and detects SBLAs in each redshift bin without global interpolation (no Pass C).
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.table import Table, vstack

C_KMS = 299792.458
LYA_REST = 1215.67


def interpolate_masked_pixels(flux, wl_obs, pixel_mask):
    """Replace flux where pixel_mask == 1 by linear interpolation from pixel_mask == 0.

    Pixels with pixel_mask == 0 are left strictly unchanged.
    """
    mask = np.asarray(pixel_mask)
    if mask.shape != flux.shape:
        raise ValueError(
            f'pixel_mask shape {mask.shape} must match flux shape {flux.shape}'
        )
    if not np.all((mask == 0) | (mask == 1)):
        raise ValueError('pixel_mask must contain only 0 (good) and 1 (masked) values')

    flux_out = np.array(flux, dtype=np.float32, copy=True)
    bad = mask == 1
    good = mask == 0

    for i in np.flatnonzero(np.any(bad, axis=1)):
        good_i = good[i]
        if not np.any(good_i):
            raise ValueError(
                f'Sightline index {i} has no unmasked (mask == 0) pixels to interpolate from'
            )
        bad_i = bad[i]
        flux_out[i, bad_i] = np.interp(
            wl_obs[bad_i], wl_obs[good_i], flux_out[i, good_i]
        ).astype(np.float32)

    return flux_out


def gaussian_sigmas(wl_array, inst_fwhm, target_fwhm):
    """Convert target and instrument FWHM (km/s) into local pixel sigma."""
    dv = C_KMS * np.gradient(wl_array) / wl_array
    eff_fwhm = np.sqrt(np.maximum(target_fwhm**2 - inst_fwhm**2, 0.0))
    return eff_fwhm / (2.354820045 * dv)


def adaptive_gaussian_filter(flux, wl_array, inst_fwhm, target_fwhm, mode='nearest'):
    """Wavelength-dependent 1D Gaussian smoothing across a 2D flux array."""
    sigma = gaussian_sigmas(wl_array, inst_fwhm, target_fwhm)
    if np.all(sigma <= 0.001):
        return flux.astype(np.float32, copy=True)

    low, high = float(sigma.min()), float(sigma.max())
    if high - low < 0.1:
        return gaussian_filter1d(flux, float(sigma.mean()), axis=1, mode=mode, output=np.float32)

    levels = np.linspace(low, high, min(10, max(3, int((high - low) / 0.1))))
    lower = np.clip(np.searchsorted(levels, sigma, side='right') - 1, 0, len(levels) - 2)
    weight_hi = np.clip((sigma - levels[lower]) / (levels[lower + 1] - levels[lower]), 0.0, 1.0)

    out = np.zeros_like(flux, dtype=np.float32)
    for idx, level in enumerate(levels):
        weight = np.where(lower == idx, 1.0 - weight_hi, 0.0)
        weight += np.where(lower + 1 == idx, weight_hi, 0.0)
        cols = np.flatnonzero(weight > 0)
        if not len(cols):
            continue
        layer = (flux if level <= 0.001
                 else gaussian_filter1d(flux, level, axis=1, mode=mode, output=np.float32))
        out[:, cols] += layer[:, cols] * weight[cols]
    return out


def velocity_footprint_bounds(wl_array, centers, velocity_kms):
    """Return half-open pixel index bounds [left, right) for velocity footprints."""
    half_wl = wl_array[centers] * velocity_kms / (2.0 * C_KMS)
    left = np.searchsorted(wl_array, wl_array[centers] - half_wl, side='left')
    right = np.searchsorted(wl_array, wl_array[centers] + half_wl, side='right')
    return left, right


def select_unmasked_centers(rows, cols, scores, wl_array, velocity_kms):
    """Greedy deepest-first center-only exclusion within one resolution scale."""
    if not len(rows):
        return np.empty(0, dtype=np.int64)

    order = np.lexsort((cols, scores, rows))
    left, right = velocity_footprint_bounds(wl_array, cols, velocity_kms)
    n_pix = len(wl_array)

    accepted = []
    current_row = -1
    blocked = None
    for idx in order:
        r, c = rows[idx], cols[idx]
        if r != current_row:
            current_row = r
            blocked = np.zeros(n_pix, dtype=bool)
        if blocked[c]:
            continue
        accepted.append(idx)
        blocked[left[idx]:right[idx]] = True
    return np.asarray(accepted, dtype=np.int64)


def count_core_covered_pixels(rows, cols, wl_seg, velocity_kms, n_los, core_mask):
    """Count unique pixels covered by footprints clipped to the unpadded core."""
    if not len(rows):
        return 0
    left, right = velocity_footprint_bounds(wl_seg, cols, velocity_kms)
    coverage = np.zeros((n_los, len(wl_seg)), dtype=bool)
    for offset in range(int(np.max(right - left))):
        target = left + offset
        inside = target < right
        coverage[rows[inside], target[inside]] = True
    return int(np.count_nonzero(coverage[:, core_mask]))


def build_z_bins(wl_obs, z_bins=None, dz=0.1):
    """Return strictly increasing redshift bin edges covering wl_obs."""
    if z_bins is not None:
        edges = np.asarray(z_bins, dtype=np.float64)
        if edges.ndim != 1 or len(edges) < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError('z_bins must be a 1D strictly increasing array with >= 2 edges')
        return edges

    z_min = wl_obs[0] / LYA_REST - 1.0
    z_max = wl_obs[-1] / LYA_REST - 1.0
    if dz <= 0:
        raise ValueError('dz must be positive')
    edges = np.arange(z_min, z_max, dz, dtype=np.float64)
    if len(edges) == 0 or (z_max - edges[-1]) > 1e-6:
        edges = np.append(edges, z_max)
    if len(edges) < 2:
        edges = np.array([z_min, z_max], dtype=np.float64)
    return edges


def find_sblas_passA_beta(
    flux,
    wl_obs,
    los_ids,
    obj_names,
    inst_fwhm_kms=0.0,
    pixel_mask=None,
    res_val_loop=(529, 340, 238, 174, 132),
    target_vf=0.004,
    z_bins=None,
    dz=0.1,
    boundary_wl=5.0,
    boundary_mode='nearest',
    deepest_first=True,
):
    """Run Pass-A-style per-redshift-bin SBLA detection (beta version).

    Parameters
    ----------
    flux : ndarray, shape (n_los, n_pix)
        Continuum-normalized flux array.
    wl_obs : ndarray, shape (n_pix,)
        Strictly increasing observed wavelength array in Angstroms.
    los_ids : array-like, shape (n_los,)
        Sightline numbers / IDs.
    obj_names : array-like, shape (n_los,)
        Object names corresponding to each sightline.
    inst_fwhm_kms : float or ndarray, shape (n_pix,)
        Instrument FWHM resolution in km/s (fixed scalar or 1D function of wavelength).
    pixel_mask : ndarray, shape (n_los, n_pix), optional
        Optional 0/1 mask array. Pixels with mask == 1 are removed and linearly
        re-interpolated from pixels with mask == 0; mask == 0 pixels are unchanged.
    res_val_loop : sequence of float
        SBLA velocity widths in km/s, ordered largest to smallest.
    target_vf : float
        Fixed target covering/volume fraction for every SBLA level (default 0.004).
    z_bins : array-like, optional
        Explicit redshift bin edges [z0, z1, ...]. Overrides dz if provided.
    dz : float
        Uniform redshift bin width used when z_bins is None (default 0.1).
    boundary_wl : float
        Wavelength padding (Angstroms) added on each side of a bin core during
        smoothing and hierarchical masking (excluded from the core VF denominator).
    boundary_mode : {'nearest', 'wrap'}
        SciPy Gaussian boundary handling at segment edges.
    deepest_first : bool
        If True, enforce deepest-first center-only exclusion within each scale.
        If False, keep all unmasked strict local minima (legacy Pass A behavior).

    Returns
    -------
    catalogue : astropy.table.Table
        Detected SBLAs across all redshift bins and velocity scales.
    bin_summary : astropy.table.Table
        Calibrated flux threshold, achieved VF, and count per (redshift bin, scale).
    """
    flux = np.asarray(flux, dtype=np.float32)
    wl_obs = np.asarray(wl_obs, dtype=np.float64)
    los_ids = np.asarray(los_ids)
    obj_names = np.asarray(obj_names)

    if flux.ndim != 2 or wl_obs.ndim != 1 or flux.shape[1] != len(wl_obs):
        raise ValueError('flux must be 2D (n_los, n_pix) matching wl_obs (n_pix,)')
    if not np.isfinite(wl_obs).all() or np.any(np.diff(wl_obs) <= 0):
        raise ValueError('wl_obs must be finite and strictly increasing')

    n_los, n_pix = flux.shape
    if len(los_ids) != n_los or len(obj_names) != n_los:
        raise ValueError('los_ids and obj_names must have length n_los')

    if np.ndim(inst_fwhm_kms) == 0:
        inst_fwhm = np.full(n_pix, float(inst_fwhm_kms), dtype=np.float64)
    else:
        inst_fwhm = np.asarray(inst_fwhm_kms, dtype=np.float64)
        if inst_fwhm.shape != (n_pix,):
            raise ValueError('inst_fwhm_kms array must have the same length as wl_obs')

    if pixel_mask is not None:
        flux = interpolate_masked_pixels(flux, wl_obs, pixel_mask)

    z_edges = build_z_bins(wl_obs, z_bins=z_bins, dz=dz)
    n_bins = len(z_edges) - 1

    cat_tables = []
    summary_rows = []

    for b_idx in range(n_bins):
        z_lo, z_hi = float(z_edges[b_idx]), float(z_edges[b_idx + 1])
        z_mid = 0.5 * (z_lo + z_hi)
        core_start = (1.0 + z_lo) * LYA_REST
        core_end = (1.0 + z_hi) * LYA_REST

        # Padded slice for smoothing context and boundary footprints
        seg_slice = (wl_obs >= core_start - boundary_wl) & (wl_obs <= core_end + boundary_wl)
        if np.count_nonzero(seg_slice) < 3:
            continue

        wl_seg = wl_obs[seg_slice]
        flux_seg = flux[:, seg_slice]
        inst_seg = inst_fwhm[seg_slice]

        # Half-open core [core_start, core_end), inclusive at the very last bin
        if b_idx == n_bins - 1:
            core_mask = (wl_seg >= core_start) & (wl_seg <= core_end)
        else:
            core_mask = (wl_seg >= core_start) & (wl_seg < core_end)

        n_core_pix = int(np.count_nonzero(core_mask))
        if n_core_pix == 0:
            continue

        total_core_pixels = n_los * n_core_pix
        # Segment-local cumulative hierarchical mask (Pass A)
        hier_mask = np.zeros((n_los, len(wl_seg)), dtype=np.int32)

        print(f"\n--- Redshift Bin {b_idx}: z=[{z_lo:.3f}, {z_hi:.3f}] "
              f"(wl={core_start:.1f}-{core_end:.1f} A, {n_core_pix} core pix) ---")

        for i_step, res_val in enumerate(res_val_loop):
            # 1. Adaptive Gaussian smoothing over the padded segment
            smoothed = adaptive_gaussian_filter(
                flux_seg, wl_seg, inst_seg, res_val, mode=boundary_mode
            )

            # 2. Strict local minima inside the unpadded core, outside prior masks
            is_min = ((smoothed[:, 1:-1] < smoothed[:, :-2])
                      & (smoothed[:, 1:-1] < smoothed[:, 2:]))
            peaks = np.pad(is_min, ((0, 0), (1, 1)), constant_values=False)
            peaks &= core_mask[None, :] & (hier_mask == 0)

            rows, cols = np.where(peaks)
            scores = smoothed[rows, cols]

            if deepest_first:
                accepted = select_unmasked_centers(rows, cols, scores, wl_seg, res_val)
                cand_rows = rows[accepted]
                cand_cols = cols[accepted]
                cand_scores = scores[accepted]
            else:
                cand_rows, cand_cols, cand_scores = rows, cols, scores

            # 3. Calibrate flux threshold in this bin via binary search to match target_vf
            if len(cand_scores) == 0:
                best_th, best_vf = np.nan, 0.0
                sel_rows = np.empty(0, dtype=np.int64)
                sel_cols = np.empty(0, dtype=np.int64)
                sel_scores = np.empty(0, dtype=np.float32)
            else:
                unique_scores = np.unique(cand_scores)

                def eval_vf(th):
                    keep = cand_scores <= th
                    covered = count_core_covered_pixels(
                        cand_rows[keep], cand_cols[keep], wl_seg, res_val, n_los, core_mask
                    )
                    return covered / total_core_pixels

                lo, hi = 0, len(unique_scores) - 1
                while lo < hi:
                    mid = (lo + hi) // 2
                    if eval_vf(unique_scores[mid]) < target_vf:
                        lo = mid + 1
                    else:
                        hi = mid

                check_indices = [lo] + ([lo - 1] if lo > 0 else [])
                best_th, best_vf, best_diff = np.nan, 0.0, float('inf')
                for idx in check_indices:
                    th = float(unique_scores[idx])
                    vf = eval_vf(th)
                    diff = abs(vf - target_vf)
                    if diff < best_diff:
                        best_th, best_vf, best_diff = th, vf, diff

                keep = cand_scores <= best_th
                sel_rows = cand_rows[keep]
                sel_cols = cand_cols[keep]
                sel_scores = cand_scores[keep]

            # 4. Update segment hierarchical mask and record detections
            if len(sel_rows) > 0:
                val_peak = int(2 ** (i_step + 1))
                val_neighbor = val_peak - 1
                left, right = velocity_footprint_bounds(wl_seg, sel_cols, res_val)
                layer = np.zeros_like(hier_mask)
                for offset in range(int(np.max(right - left))):
                    target = left + offset
                    inside = target < right
                    layer[sel_rows[inside], target[inside]] = val_neighbor
                layer[sel_rows, sel_cols] = val_peak
                update = (hier_mask == 0) & (layer != 0)
                hier_mask[update] = layer[update]

                wl_abs = wl_seg[sel_cols]
                half_wl = wl_abs * res_val / (2.0 * C_KMS)
                cat_tables.append(Table({
                    'OBJ_NAME': obj_names[sel_rows],
                    'LOS_ID': los_ids[sel_rows],
                    'Z_BIN': np.full(len(sel_cols), b_idx, dtype=np.int32),
                    'LAMBDA_ABS': wl_abs,
                    'Z': wl_abs / LYA_REST - 1.0,
                    'VEL_SBLA': np.full(len(sel_cols), res_val, dtype=np.int64),
                    'LAMBDA_MIN': wl_abs - half_wl,
                    'LAMBDA_MAX': wl_abs + half_wl,
                    'FLUX_SMOOTH': sel_scores,
                    'FLUX_RAW': flux_seg[sel_rows, sel_cols],
                    'FLUX_THRESHOLD': np.full(len(sel_cols), best_th, dtype=np.float64),
                }))

            summary_rows.append({
                'Z_BIN': b_idx,
                'Z_MIN': z_lo,
                'Z_MAX': z_hi,
                'Z_MID': z_mid,
                'VEL_SBLA': res_val,
                'FLUX_THRESHOLD': best_th,
                'TARGET_VF': target_vf,
                'ACTUAL_VF': best_vf,
                'N_DETECTED': len(sel_rows),
            })
            print(f"  Res {res_val:3d} km/s | F_th = {best_th:+.4f} | "
                  f"VF = {best_vf:.5f} (target {target_vf:.5f}) | N = {len(sel_rows)}")

    if cat_tables:
        catalogue = vstack(cat_tables)
    else:
        catalogue = Table(names=(
            'OBJ_NAME', 'LOS_ID', 'Z_BIN', 'LAMBDA_ABS', 'Z', 'VEL_SBLA',
            'LAMBDA_MIN', 'LAMBDA_MAX', 'FLUX_SMOOTH', 'FLUX_RAW', 'FLUX_THRESHOLD'
        ))
    bin_summary = Table(summary_rows)
    return catalogue, bin_summary


# ==========================================
# Quick Beta Self-Test
# ==========================================
if __name__ == '__main__':
    rng = np.random.default_rng(42)

    n_los = 200
    wl_obs = np.arange(3700.0, 4500.0, 0.8)  # z ~ 2.04 to 2.70
    n_pix = len(wl_obs)

    los_ids = np.arange(1000, 1000 + n_los)
    obj_names = np.array([f'BETA_QSO_{i:04d}' for i in range(n_los)])

    # Instrument resolution: scalar (e.g., 75.0) OR 1D array of length n_pix
    inst_fwhm_kms = np.linspace(90.0, 65.0, n_pix)

    # Synthetic Ly-alpha forest beta spectra
    flux = np.ones((n_los, n_pix), dtype=np.float32)
    for i in range(n_los):
        n_lines = rng.integers(15, 35)
        centers = rng.uniform(wl_obs[0], wl_obs[-1], size=n_lines)
        widths = rng.uniform(0.6, 5.0, size=n_lines)
        taus = rng.exponential(scale=1.2, size=n_lines)
        tau_tot = np.zeros(n_pix)
        for c0, w0, t0 in zip(centers, widths, taus):
            tau_tot += t0 * np.exp(-0.5 * ((wl_obs - c0) / w0) ** 2)
        flux[i] = np.exp(-tau_tot) + rng.normal(0.0, 0.05, size=n_pix)

    # Optional pixel mask: 0 = good (unchanged), 1 = masked (removed & re-interpolated)
    pixel_mask = np.zeros((n_los, n_pix), dtype=np.uint8)
    pixel_mask[:, 200:205] = 1
    flux_corrupted = flux.copy()
    flux_corrupted[pixel_mask == 1] = -999.0

    # Verify mask == 0 pixels are strictly unchanged
    flux_interp = interpolate_masked_pixels(flux_corrupted, wl_obs, pixel_mask)
    assert np.array_equal(flux_interp[pixel_mask == 0], flux_corrupted[pixel_mask == 0])

    catalogue, bin_summary = find_sblas_passA_beta(
        flux=flux_corrupted,
        wl_obs=wl_obs,
        los_ids=los_ids,
        obj_names=obj_names,
        inst_fwhm_kms=inst_fwhm_kms,      # scalar or 1D array
        pixel_mask=pixel_mask,            # optional; pass None to disable
        res_val_loop=[529, 340, 238, 174, 132],
        target_vf=0.004,
        dz=0.1,                           # or pass explicit z_bins=[2.1, 2.2, ...]
        boundary_wl=5.0,
    )

    print("\nPer-bin flux limits and summary (first 15 rows):")
    print(bin_summary[:15])
    print("\nDetected SBLAs (first 10 rows):")
    print(catalogue[:10])
    # catalogue.write('SBLA_Beta_Catalogue.fits', overwrite=True)
    # bin_summary.write('SBLA_Beta_FluxLimits.fits', overwrite=True)
