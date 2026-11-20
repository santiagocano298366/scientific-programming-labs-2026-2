# preprocessing

From the raw file to the usable curves.

| File | Content |
|---|---|
| `loading.py` | `load_raw`: reads the csv file. |
| `cleaning.py` | `clean_points`, `group_textures`, `build_curve_table`, `select_usable_curves`, `flag_upward_jumps`, `build_curve_data`. |
| `optimized.py` | `build_curve_data_groupby`: the same as `build_curve_data` with one groupby. |
| `complexity_analysis.ipynb` | Complexity analysis, optimization and times. |

Steps: remove duplicates, keep `data_flag == "good quality estimate"`, remove missing values, remove psi > 100000 m, check the physical range, keep the tropical climate, group the textures, make one row per curve, keep the curves with at least 6 points that cover psi <= 1 m and psi >= 100 m, and mark the curves where theta rises more than 0.05.
