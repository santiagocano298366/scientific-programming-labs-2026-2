# data

* `raw/`: original file, not modified. GSHP (Global Soil Hydraulic Properties) dataset, Gupta et al. (2022), 136,989 rows and 54 columns, encoding latin-1. Each row is a measured point (psi, theta) and each curve (`layer_id`) has several rows.
* `processed/`: files saved by `main.py`: the cleaning steps, the usable curves, their points and the table of features.

Main columns of the raw file: `layer_id` (curve), `profile_id` (site), `reference` (source), `climate_classes`, `tex_psda` (USDA texture), `lab_head_m` (psi in m), `lab_wrc` (theta in m3/m3), `alpha` and `n` (published van Genuchten parameters), `data_flag` (quality of the fit made by the authors).
