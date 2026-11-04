# Basis maps

`registry.json` is the record of everything the library contains and how to rebuild it. One recipe covers the whole
library (`mask_path`, `transform`); each cohort contributes one entry and, once built, exactly one shard.

For each image column in `voxelwise_vars` the builder reads the images once and runs one multi-output regression,
`regressand_1 + regressand_2 + ... ~ image + covariates`, with `scripts/regression_pipeline.py`'s table preparation and
regression engine. Every symptom becomes one basis named `{regressand}-on-{image}`: the t map of the image's coefficient.
With `drop_nans` (the default) a patient missing any symptom, the image or a covariate is dropped, so every symptom is fitted on
the same patients in one regression. Fill or remove symptoms in the table when that leaves too few complete rows.

```json
{
  "mask_path": "/path/to/MNI152_T1_2mm_brain_mask.nii",
  "transform": "rank",
  "cohorts": {
    "example_cohort": {
      "pathology": "Stroke",
      "input_path": "/path/to/table.csv",
      "subject_column": "subject_id",
      "regressand_list": ["gait_score", "speech_score"],
      "voxelwise_vars": ["lesion_path", "lnm_path"],
      "drop_rows": [["selected", "equal", 0]]
    }
  }
}
```

`regressand_list` and `voxelwise_vars` may be tuples in Python registries; they are arrays in JSON.
Fiber columns (`*.fib.npy` / `*.fib.json`) are regressed against the registry's top-level `fiber_atlas`, the atlas their values index, exactly as `regression_pipeline.py` does with that atlas as its mask. Each fiber t map is then rasterized onto `mask_path` the way the pipeline writes its fiber NIfTI (`TractDensity`, every fiber, no threshold, no mirroring), so every basis in a shard is a voxel map.
Required in every entry: `pathology`, `input_path`, `subject_column`, `regressand_list`, `voxelwise_vars`.
Optional, with the defaults of `build.load_table` and `build.fit_bases`: `sheet`, `drop_rows`, `one_hot`, `invert_regressand`
(negates every regressand), `drop_nans`, `covariates_list`, `add_intercept`, `weights_col`. Rows are prepared in the order
column cleaning, inversion, row drops, one-hot, then NaN dropping per regression, so one-hot columns can be covariates.
Column names are the names after the pipeline's column cleaning. Permutations, cross-validation and exchangeability blocks
do not change a basis and are not part of an entry.

Build every cohort that has no shard yet with
`python -m calvin_utils.neuroimaging_utils.ccm_utils.optimization.basis_maps.build`, or call `build.main(registry, directory)`
with a study-specific registry and output directory.

Each built cohort is two files:

- `shard_{cohort}.npy` — float32, shape `(voxels, bases)`, one column per basis in `voxelwise_vars` x `regressand_list` order, in the mask's voxel order. Read only through `np.load(path, mmap_mode="r")`.
- `shard_{cohort}.json` — the entry with every default filled in, plus `cohort`, `mask_path`, `transform`, `patient_ids` (every subject used by any basis), `n_voxels`, `created`, and per basis `name`, `regressand`, `voxelwise_var` and `n`.

An existing shard whose sidecar no longer matches its registry entry stops the build; delete both files to rebuild that cohort.
`pipeline/basis_damage.py` validates both files whenever shards are loaded.

The library is fitted once, on every patient in each cohort; cross-validation refits only the weights against it.
