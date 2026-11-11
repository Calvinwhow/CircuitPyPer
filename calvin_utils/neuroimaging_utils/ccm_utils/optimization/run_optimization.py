"""Symptom-basis optimization: run() reads top to bottom as the steps of one analysis, and each step is one module."""

import hashlib
from pathlib import Path
from . import algorithms
from .inference import basis_map
from .neural_network import fitting as network_fitting
from .output import saving
from .pipeline import cross_validation, pre_processing, prediction
from .pipeline.damage_matrix_factory import DamageMatrixFactory
from .pipeline.damage_matrix_utils.basis_library import BASIS_DIR
from .pipeline.model_fitter import ModelFitter
from .pipeline.model_fitter_utils.model_fitter_schema import AlgorithmResult, ModelFitterConfig, ModelFitterDependencies
from .projections import prediction_variance
from .selection import prepare_regularization
from .solvers import adam, proximal_gradient, semidefinite


class RunOptimization:
    """
    This is an optimization algorithm which takes a set of symptom-basis maps and finds the combination ofthem which best explain outcomes. 
        def: Symptom-basis map represent lesion network maps, VLSM maps, voxelized fiber-symptom maps, or additional localizations like gene-symptom maps)
    
    The overall process is simple: score patients against the basis library (how much does a patient 'hit' the localization) and then find the localizations which best establish a monotonic relationship between 'hitting' the anatomy and having (or treating) the symptom.
    Orchestrates symptom-basis scoring, cross-validation, prediction across multiple algorithms, a neural network, merging maps, saving the merged map, and spitting out decomposed versions of the individual symptom-map domains which summate to form the final map (i.e. contributing fibers, contributing network, etc.)
    Represents the generalized form of convergent causal mapping across localizations, cohorts, and symptoms. 
    """

    def __init__(self, damage, *, methods=algorithms.__all__, sparse_semidefinite=False, neural_network=False, fully_trainable=False, weight_mode="unweighted", dataset_weights=None, dataset_signs=None, regularization_strength=0.1, project_prediction_variance=False, learning_rate=0.01, max_epochs=2000, tolerance=1e-7, patience=100, weight_decay=0.0, gradient_clip=5.0, device="auto", seed=42):
        """Configure optimization over one typed damage matrix.

        Stored attributes:
            damage: DamageMatrix | scalar object | canonical fitting and test damage with symptom-basis maps.
            dataset_weights: mapping[str, float] or None | length active dataset count | optional contribution weight for each dataset.
            seed: int | scalar | reproducible seed for folds and stochastic optimizers.
            model_fitter: ModelFitter | scalar object | canonical typed fitting entry point.
        """
        self.damage = damage
        self.dataset_weights = dataset_weights
        self.seed = seed
        selected_methods = tuple(method for method in methods if sparse_semidefinite or method != "sparse_semidefinite")
        config = ModelFitterConfig(
            methods=selected_methods,
            neural_network=neural_network,
            fully_trainable=fully_trainable,
            weight_mode=weight_mode,
            dataset_weights=dataset_weights,
            dataset_signs=dataset_signs,
            regularization_strength=regularization_strength,
            project_prediction_variance=project_prediction_variance,
            learning_rate=learning_rate,
            max_epochs=max_epochs,
            tolerance=tolerance,
            patience=patience,
            weight_decay=weight_decay,
            gradient_clip=gradient_clip,
            device=device,
        )
        dependencies = ModelFitterDependencies(
            algorithms=algorithms,
            adam_solver=adam.main,
            proximal_solver=proximal_gradient.main,
            semidefinite_solver=semidefinite.main,
            prediction_variance=prediction_variance,
            calibrate_regularization=prepare_regularization.main,
            dataset_weights=pre_processing.dataset_weights,
            initial_weights=pre_processing.initial_weights,
            network_fit=network_fitting.main,
        )
        self.model_fitter = ModelFitter(damage, config, dependencies)

    def run(self, *, cross_validation=None, out_dir=None, output_type=None, mask_path=None, store_history=False, store_map_stack=False, render_gif=False, gif_fps=5, gif_max_frames=60, gif_view="auto", winner_only=False, save_anatomy_maps=False, n_permutations=0):
        """
        cross_validation: "lodo" | "loocv" | int | None; leave-one-dataset-out, pooled patient folds across every dataset, or one fit on every patient (None), the only fit with a map to test and regress.
        out_dir, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, winner_only, save_anatomy_maps: outputs. n_permutations: basis-map regression of the winner.
        """
        self.fit(cross_validation)
        self.predict()
        self.rank()
        self.save(out_dir, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, winner_only, save_anatomy_maps)
        self.infer(output_type, mask_path, n_permutations)
        return self.comparison

    def fit(self, scheme):
        self.results = cross_validation.main(self.damage, scheme, self.model_fitter, self.seed)

    def predict(self):
        self.results = prediction.main(self.results, self.damage)

    def rank(self):
        self.comparison = prediction.rank(self.results)
        self.result = self.results.get(self.comparison.winner)

    def save(self, out_dir, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, winner_only, save_anatomy_maps):
        self.saved = saving.main(self.results, self.damage, out_dir, self.comparison.winner, winner_only, output_type, mask_path, store_history, store_map_stack, render_gif, gif_fps, gif_max_frames, gif_view, save_anatomy_maps)

    def infer(self, output_type, mask_path, n_permutations):
        self.result.basis_map_regression = basis_map.infer(self.result, self.saved.get(self.comparison.winner), output_type, mask_path, n_permutations)

    def gradient_step(self, gradient, learning_rate=0.01):
        """Exposed so an external optimizer can drive the winning algorithm: one gradient-ascent step on its weights, returning its native merit."""
        if not isinstance(self.result, AlgorithmResult):
            raise TypeError("Gradient steps require an algorithm result, not a cross-validation or neural-network result.")
        self.result = self.model_fitter.step(self.result, gradient, learning_rate)
        self.results = self.results.replace(self.result)
        prediction.main(self.results, self.damage)
        return self.model_fitter.merit(self.result)

    @classmethod
    def run_targets(cls, dataset_specs, targets=None, *, test_specs=None, subject_col=None, cohorts=None, basis_dir=BASIS_DIR, cross_validation="lodo", out_dir=None, output_type=None, mask_path=None, store_history=False, store_map_stack=False, render_gif=False, gif_fps=5, gif_max_frames=60, gif_view="auto", winner_only=False, save_anatomy_maps=False, n_permutations=0, **settings):
        """Run once per top-level target, with nested tuples optimized jointly.

        ``("gait", "speech")`` performs two independent optimizations.
        ``(("gait", "speech"),)`` performs one optimization with one shared map
        fitted jointly to both outcomes. Mixed independent and joint groups are
        allowed. When targets is None, every basis symptom runs independently.

        dataset_specs, test_specs: {name: {"csv_path", "nifti_column", "subject_column"}}.
        """
        factory = DamageMatrixFactory(cohorts, basis_dir)
        if targets is None:
            targets = factory.basis_library.symptoms
        groups = cls._target_groups(targets)
        columns = tuple(dict.fromkeys(target for group in groups for target in group))
        fitting_cohorts = pre_processing.load_target_cohorts(dataset_specs, columns, mask_path, subject_col)
        test_cohorts = cls._load_test_cohorts(test_specs, columns, mask_path, subject_col, cross_validation)
        all_damage = factory(fitting_cohorts, test_cohorts)
        comparisons = {}
        for index, group in enumerate(groups, start=1):
            label = cls._target_group_label(group)
            if len(group) == 1:
                print(f"Target {index}/{len(groups)}: {group[0]}", flush=True)
            else:
                print(f"Joint target {index}/{len(groups)} ({len(group)} outcomes): {', '.join(group)}", flush=True)
            group_damage = all_damage.select(group)
            group_out_dir = cls._target_output_directory(out_dir, label)
            optimizer = cls(group_damage, **settings)
            comparisons[label] = optimizer.run(cross_validation=cross_validation, out_dir=group_out_dir, output_type=output_type, mask_path=mask_path, store_history=store_history, store_map_stack=store_map_stack, render_gif=render_gif, gif_fps=gif_fps, gif_max_frames=gif_max_frames, gif_view=gif_view, winner_only=winner_only, save_anatomy_maps=save_anatomy_maps, n_permutations=n_permutations)
        saving.target_validation(comparisons, out_dir)
        return comparisons

    @staticmethod
    def _target_groups(targets):
        """Normalize the public target syntax to nonempty tuples of column names."""
        if isinstance(targets, str):
            targets = (targets,)
        groups = []
        for entry in tuple(targets):
            group = RunOptimization._normalize_target_entry(entry)
            if not group or any(not isinstance(target, str) or not target for target in group):
                raise ValueError("Each optimization target must be a column name or a nonempty tuple of column names.")
            if len(set(group)) != len(group):
                raise ValueError(f"Joint target group repeats a column: {group}.")
            groups.append(group)
        if not groups:
            raise ValueError("At least one optimization target is required.")
        return tuple(groups)

    @staticmethod
    def _normalize_target_entry(entry):
        """Resolve one string or nested target collection to one target tuple."""
        if isinstance(entry, str):
            return (entry,)
        return tuple(entry)

    @staticmethod
    def _load_test_cohorts(test_specs, columns, mask_path, subject_col, cross_validation):
        """Load external tests only for the final fit on every patient."""
        if not test_specs:
            return None
        if cross_validation is not None:
            return None
        return pre_processing.load_target_cohorts(test_specs, columns, mask_path, subject_col)

    @staticmethod
    def _target_output_directory(out_dir, label):
        """Resolve an optional analysis root to this target's output directory."""
        if out_dir is None:
            return None
        return Path(out_dir) / label

    @staticmethod
    def _target_group_label(group):
        """Return a readable, filesystem-safe-enough directory label for a target group."""
        if len(group) == 1:
            return group[0]
        label = "joint__" + "__".join(group)
        if len(label.encode()) <= 220:
            return label
        digest = hashlib.sha256("\0".join(group).encode()).hexdigest()[:12]
        return f"joint__{group[0]}__{len(group)}_targets__{digest}"
