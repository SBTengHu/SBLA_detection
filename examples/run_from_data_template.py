#!/usr/bin/env python
# coding: utf-8

"""Template for running SBLA Finder (Beta — Pass A) on your own dataset.

Replace the placeholder file paths and column/key names below with your own
FITS, HDF5, or NPZ inputs.
"""

from pathlib import Path
import sys
import numpy as np
from astropy.io import fits

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from sbla_beta_passA import find_sblas_passA_beta


def load_your_data():
    """Replace this loader with your own FITS/HDF5/NPZ reading logic."""
    # Example FITS layout:
    with fits.open("path/to/your_spectra.fits") as hdul:
        flux = hdul["FLUX"].data              # shape: (n_los, n_pix)
        wl_obs = hdul["WAVELENGTH"].data      # shape: (n_pix,) in Angstroms
        los_ids = hdul["META"].data["LOS_ID"] # shape: (n_los,)
        obj_names = hdul["META"].data["NAME"] # shape: (n_los,)

        # Optional: 0 = good, 1 = masked (set to None if not used)
        pixel_mask = hdul["MASK"].data if "MASK" in hdul else None

        # Instrument FWHM in km/s: scalar float OR 1D array matching wl_obs
        inst_fwhm_kms = 75.0

    return flux, wl_obs, los_ids, obj_names, inst_fwhm_kms, pixel_mask


def main():
    flux, wl_obs, los_ids, obj_names, inst_fwhm_kms, pixel_mask = load_your_data()

    catalogue, bin_summary = find_sblas_passA_beta(
        flux=flux,
        wl_obs=wl_obs,
        los_ids=los_ids,
        obj_names=obj_names,
        inst_fwhm_kms=inst_fwhm_kms,
        pixel_mask=pixel_mask,
        res_val_loop=[529, 340, 238, 174, 132],
        target_vf=0.004,
        dz=0.1,             # or pass z_bins=[2.0, 2.2, 2.4, 2.6]
        boundary_wl=5.0,
        boundary_mode="nearest",  # use "wrap" for periodic simulation skewers
    )

    out_dir = REPO_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    catalogue.write(out_dir / "SBLA_Beta_Catalogue.fits", overwrite=True)
    bin_summary.write(out_dir / "SBLA_Beta_FluxLimits.fits", overwrite=True)


if __name__ == "__main__":
    main()
