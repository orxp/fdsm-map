# Public example data

The repository overview image is generated from two public analytical signals.
Raw downloads are excluded from Git and are fetched only when
`examples/make_toc_figure.py` is run. Each download is pinned by SHA-256.

| Modality | Signal | Source | Local processing |
| --- | --- | --- | --- |
| GC–MS | TIC from `EFWS-1.mzML` in MetaboLights study MTBLS11550 | [EMBL-EBI file](https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS11550/FILES/DERIVED_FILES/GC/EFWS-1.mzML) | The TIC arrays are decoded from mzML and linearly resampled to an equal-length uniform time grid before FDSM transformation. |
| FT–IR | `ftir_ldpe_soil.asp`, distributed with OpenSpecy | [OpenSpecy source file](https://github.com/wincowgerDEV/OpenSpecy-package/blob/main/inst/extdata/ftir_ldpe_soil.asp) | The wavenumber and intensity arrays are read from the ASP file. The descending spectral axis is handled by `fdsm.transform`. |

The OpenSpecy package repository is licensed under
[CC BY 4.0](https://github.com/wincowgerDEV/OpenSpecy-package/blob/main/LICENSE.md).
MetaboLights explains public study data access and reuse in its
[data access guide](https://www.ebi.ac.uk/training/online/courses/metabolights-quick-tour/getting-data-from-metabolights/);
users should also consult the [EMBL-EBI terms of use](https://www.ebi.ac.uk/about/terms-of-use/).
These source datasets are not covered by this repository's MIT software
license.

Pinned source hashes:

- `EFWS-1.mzML`: `f3ead5cdae2ed4be96e8b5b8514da3b451ac7787f54c3fcebd5aab28e3727b2d`
- `ftir_ldpe_soil.asp`: `96d2eea61cdc3ba8f675fde1566388949c2f43689b4e3eb6d3d41457df015547`

## Reproduce the image

From an environment where `fdsm-map[plot]` is installed:

```bash
python examples/make_toc_figure.py
```

This creates `docs/assets/fdsm_1d_to_2d.png`. Both transforms use derivative
orders 0–2 in increments of 0.125 and a Savitzky–Golay window of 11 samples.
The plotted maps use the library's default per-order z-score normalization.
The downloaded TIC contains 3,373 observations from 3.00035 to 30.00567 min.
Its acquisition intervals include gaps, so the example linearly resamples it to
3,373 uniform points (0.00800869 min spacing) before applying the FFT-based
transform. The FT–IR file contains 1,798 uniformly spaced points from 3999.4335
to 650.4205 cm⁻¹; its descending axis is reversed internally with the signal.
