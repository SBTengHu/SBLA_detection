# SBLA Finder (Beta — Pass A)

A lightweight, single-bin Python implementation for detecting **Strong Blended Lyman-α Absorbers (SBLAs)** in Lyα forest spectra using hierarchical, multi-scale adaptive Gaussian smoothing.

This **beta** version is designed for rapid testing, mock datasets, and uniform-SNR samples. It implements the per-redshift-bin threshold calibration (**Pass A**) directly, omitting SNR binning and global threshold interpolation (**Pass B/C**) from the full survey pipeline.

## Key Features

- **Top-Down Hierarchical Detection:** Iterates from wide to narrow velocity smoothing scales (default `529, 340, 238, 174, 132 km/s`), cumulatively masking detected footprints so smaller-scale absorbers do not re-trigger inside larger systems.
- **Adaptive Instrument Resolution:** Accepts `inst_fwhm_kms` as either a fixed scalar (km/s) or a 1D wavelength-dependent array, smoothing each pixel in quadrature ($\sqrt{\max(v_{\mathrm{target}}^2 - v_{\mathrm{inst}}^2, 0)}$).
- **Fixed Target Volume Fraction:** Calibrates the optimal smoothed flux threshold ($F_{\mathrm{th}}$) in each redshift bin via binary search to match a fixed target covering/volume fraction (`target_vf = 0.004` by default).
- **Optional Pixel Masking & Re-Interpolation:** If a `0/1` `pixel_mask` is supplied, flux pixels with `mask == 1` are removed and linearly re-interpolated from `mask == 0` pixels; unmasked (`mask == 0`) flux values remain strictly unchanged.

## Simplifications in Beta Version

- **Single SNR bin:** No SNR computation, moving SNR window, or SNR cuts are applied.
- **No Pass B/C interpolation:** Flux limits are calibrated and applied directly inside each redshift bin (`z_bins` or uniform `dz`) with padded smoothing margins (`boundary_wl`).
- **All pixels assumed valid:** No continuum/IVAR/DLA quality cuts are applied beyond optional `pixel_mask` interpolation.

---

## Installation

Python 3.9+ is recommended.

```bash
git clone [https://github.com/](https://github.com/)<your-username>/sbla-finder-beta.git
cd sbla-finder-beta
pip install -r requirements.txt
