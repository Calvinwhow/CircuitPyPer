# Anatomical neural integration

`RunNeuralNetwork` fits the five full-basis optimization methods, orients their
coefficient vectors on the fitting data, and passes the shared coefficient
matrix through fibers, networks, and local anatomy.

The code is separated by responsibility:

- `induction.py`: five full-basis deterministic fits and coefficient orientation.
- `anatomy.py`: representation metadata, masks, normalization, and reconstruction.
- `layers.py`: the three explicit spatial equations `Z`, `H`, and `M`.
- `model.py`: readable forward composition of those three layers and patient scoring.
- `losses.py`: pair construction and the shared pairwise softplus loss.
- `training.py`: float32 training, prediction-variance normalization, and checkpointing.
- `saving.py`: portable maps, parameters, predictions, and validation results.

Each branch reconstructs five voxel maps, applies an algorithm-specific bias
and ReLU, mixes the algorithms, and applies a second bias and ReLU. Three learned
weights combine the branch maps, each minus a learned baseline, into the final
localization map. Those three weighted branch maps sum exactly to the final map.
Patient maps are scored only after these spatial nonlinearities.

Initialization starts at the linear solution. Each algorithm bias starts at minus
its map's minimum and each branch bias at zero, so both ReLUs pass the initial maps
through unchanged; each baseline starts at the constant those biases add, so the
initial map is exactly the alpha- and beta-weighted sum of the algorithm maps.
Training then learns thresholds from there. A baseline acts on every voxel, so it
adds a term proportional to the patient map's sum to every score: an explicit
lesion-extent term reported in `integration_weights.csv`.

```python
from calvin_utils.neuroimaging_utils.ccm_utils.optimization import RunNeuralNetwork

runner = RunNeuralNetwork(
    datasets,
    basis_dir="/path/to/basis_maps",
    basis_representations={
        "cohort_a:basis_0001": "fibers",
        "cohort_a:basis_0002": "networks",
        "cohort_a:basis_0003": "local",
    },
    mode="frozen",
)
result = runner.run(out_dir="/path/to/output", output_type="nii", mask_path="/path/to/mask.nii.gz")
```

The representation label should normally be written into each basis sidecar by
adding `representation` to its basis registry entry. A mapping passed to the
runner is supported for existing basis libraries. Labels are never inferred
from filenames.

Modes:

- `frozen`: reconstruct the 15 maps once and train only 39 integration scalars.
- `fine_tune`: initialize from all five deterministic solutions and train the
  shared basis-by-algorithm coefficient matrix as well.
- `scratch`: randomly initialize and train that shared coefficient matrix.

Patient maps are always L2-normalized, as in the algorithms. Validation
refits both the deterministic algorithms and neural integration on each
training split.

Training uses full-batch Adam. Every fitting patient and every non-tied
clinical pair contributes to every epoch; there is no patient or pair
mini-batching. Early stopping observes training loss only and never reads a
held-out fold or test cohort.

The network is one model in `RunOptimization(..., neural_network=True)`, compared with every
algorithm in `methods` on the same splits by `pipeline/cross_validation.py`:

- `cross_validation="lodo"` fits D-1 cohorts and predicts the omitted cohort, once per cohort.
- `cross_validation=k` or `"loocv"` holds out part of every cohort per fold and fits the rest jointly.
- `test_datasets=...` is an explicit external test set. It is never used for
  orientation, neural fitting, normalization, or early stopping.

Each fold fits the algorithms, then the network on those same solutions. With
`cross_validation=None` every model is fitted once on all fitting cohorts; only that
fit is scored on the external test set and passed to basis-map regression.
