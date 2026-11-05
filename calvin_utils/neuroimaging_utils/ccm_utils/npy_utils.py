from scipy.stats import rankdata
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

_DEFAULT_MMAP_MODE = object()

class DataLoader:
    def __init__(self, data_dict_path, mmap_mode=None):
        """Load packaged arrays, optionally as NumPy read-only memory maps.

        ``mmap_mode='r'`` keeps large .npy arrays on disk. Individual loads can
        override this default with ``load_dataset(..., mmap_mode=None)``.
        """
        self.data_dict_path = data_dict_path
        self.mmap_mode = mmap_mode
        with open(data_dict_path, 'r') as f:
            self.dataset_paths_dict = json.load(f)
        self.dataset_names_list = [k for k in self.dataset_paths_dict.keys()]

    @classmethod
    def from_csv_dict(cls, dataset_specs, out_dir, mask_path=None,
                      subject_col=None, mmap_mode=None,
                      data_transform_method='standardize'):
        """Package CSV symptom specifications with ``RegressionPrep``.

        Each entry must define ``csv_path``, ``nifti_column``, and
        ``symptom_column``. ``subject_column`` and ``sheet`` are optional.
        ``RegressionPrep`` owns image loading and all rank/standardize behavior;
        this adapter only writes the small DataLoader manifest and patient IDs.
        """
        required = {'csv_path', 'nifti_column', 'symptom_column'}
        if not dataset_specs or not isinstance(dataset_specs, dict):
            raise ValueError("Provide a nonempty CSV dataset dictionary.")
        normalized = []
        for name, spec in dataset_specs.items():
            if not isinstance(spec, dict) or not required <= set(spec):
                raise ValueError(
                    f"Dataset {name!r} needs csv_path, nifti_column, and "
                    "symptom_column; raw X/y/ids arrays are not accepted."
                )
            if any(char in str(name) for char in ('/', '\\')):
                raise ValueError(f"Dataset name cannot contain a path separator: {name!r}")
            table_path = Path(spec['csv_path']).expanduser().resolve()
            if not table_path.is_file():
                raise FileNotFoundError(f"Patient table does not exist: {table_path}")
            sheet = spec.get('sheet')
            frame = (
                pd.read_excel(table_path, sheet_name=sheet if sheet is not None else 0)
                if table_path.suffix.lower() in {'.xlsx', '.xls'}
                else pd.read_csv(table_path)
            )
            image_col, symptom = spec['nifti_column'], spec['symptom_column']
            id_col = spec.get('subject_column', subject_col)
            missing = ({image_col, symptom} | ({id_col} if id_col else set())) - set(frame.columns)
            if missing:
                raise ValueError(f"{table_path} is missing columns: {sorted(missing)}")
            outcome = pd.to_numeric(frame[symptom], errors='coerce')
            if (frame[symptom].notna() & outcome.isna()).any():
                raise ValueError(f"Outcome {symptom!r} contains nonnumeric values.")
            valid = frame[image_col].notna() & outcome.notna()
            if id_col:
                valid &= frame[id_col].notna()
            rows = frame.loc[valid]
            image_paths = []
            for value in rows[image_col]:
                path = Path(str(value).strip()).expanduser()
                path = path if path.is_absolute() else table_path.parent / path
                path = path.resolve()
                if not path.is_file():
                    raise FileNotFoundError(f"Patient image does not exist: {path}")
                image_paths.append(str(path))
            ids = (
                rows[id_col].astype(str).str.strip().tolist() if id_col
                else [f"{table_path}:{index}" for index in rows.index]
            )
            if len(rows) < 5 or outcome.loc[valid].nunique() < 2:
                raise ValueError(f"Dataset {name!r} needs five rows and a varying symptom.")
            if len(set(ids)) != len(ids) or any(not value for value in ids):
                raise ValueError(f"Dataset {name!r} needs unique nonempty patient IDs.")
            normalized.append(pd.DataFrame({
                '__dataset__': str(name), '__nifti__': image_paths,
                '__symptom__': outcome.loc[valid].to_numpy(float), '__id__': ids,
            }))

        from calvin_utils.permutation_analysis_utils.voxelwise_regression_prep import RegressionPrep

        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        combined = pd.concat(normalized, ignore_index=True)
        manifest = {}
        output_types = set()
        for name, rows in combined.groupby('__dataset__', sort=False):
            dataset_dir = out_dir / str(name)
            preparer = RegressionPrep(
                design_matrix=rows[['__symptom__']].reset_index(drop=True),
                contrast_matrix=np.ones((1, 1)),
                outcome_df=rows[['__nifti__']].reset_index(drop=True),
                out_dir=str(dataset_dir),
                neuroimaging_variables=['__nifti__'],
                mask_path=mask_path,
                data_transform_method=data_transform_method,
            )
            regression_manifest, _ = preparer.run()
            payload = regression_manifest['neuroimaging_regression']
            ids_path = dataset_dir / 'ids.npy'
            np.save(ids_path, rows['__id__'].to_numpy(dtype='U'))
            manifest[str(name)] = {
                'niftis': payload['outcome_data'],
                'indep_var': payload['design_matrix'],
                'ids': str(ids_path),
            }
            output_types.add(payload['output_ftype'])
        if len(output_types) != 1:
            raise ValueError(f"CSV datasets use mixed image types: {sorted(output_types)}")
        manifest_path = out_dir / 'dataset_dict.json'
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
        loader = cls(manifest_path, mmap_mode=mmap_mode)
        loader.output_type = output_types.pop()
        return loader
        
    def load_dataset(self, dataset_name, nifti_type='niftis', mmap_mode=_DEFAULT_MMAP_MODE):
        """
        Loads data for `dataset_name`. By default, returns raw 'niftis'.
        If `nifti_type='niftis_ranked'`, ensures 'niftis_ranked' exist, 
        creating them if necessary, then loads them.
        """
        paths = self.dataset_paths_dict[dataset_name]
        if mmap_mode is _DEFAULT_MMAP_MODE:
            mmap_mode = self.mmap_mode
        
        if nifti_type == 'niftis_ranked':
            self._init_ranked_niftis(dataset_name)
        
        niftis = np.load(paths[nifti_type], mmap_mode=mmap_mode)
        indep_var = np.load(paths['indep_var'], mmap_mode=mmap_mode)
        if niftis.ndim == 3 and niftis.shape[1] == 1:
            niftis = niftis[:, 0, :]
        if indep_var.ndim == 3 and indep_var.shape[2] == 1:
            indep_var = indep_var[:, :, 0]
        data = {'niftis': niftis, 'indep_var': indep_var}
        if 'covariates' in paths:
            data['covariates'] = np.load(paths['covariates'], mmap_mode=mmap_mode)
        return data
    
    @staticmethod
    def load_dataset_static(data_paths_dict, dataset_name, mmap_mode=None):
        paths = data_paths_dict[dataset_name]

        data_dict = {
            'niftis': np.load(paths['niftis'], mmap_mode=mmap_mode),
            'indep_var': np.load(paths['indep_var'], mmap_mode=mmap_mode),
        }
        if 'covariates' in paths:
            data_dict['covariates'] = np.load(paths['covariates'], mmap_mode=mmap_mode)
        return data_dict
    
    def _rank_niftis(self, arr, vectorize=True):
        """
        Rank a 2D array (voxels, samples) ignoring ties.
        This is a 'sloppy' approach: ties get consecutive ranks arbitrarily.
        """
        if vectorize: 
            ranks = np.empty_like(arr, dtype=float)
            rows = np.arange(arr.shape[0])[:, None]
            cols = np.arange(arr.shape[1])
            idx_sorted = np.argsort(arr, axis=0)
            ranks[idx_sorted, cols] = rows
        else: 
            ranks = np.apply_along_axis(rankdata, 0, arr) 
        return ranks
    
    def _init_ranked_niftis(self, dataset_name):
        """
        Internal helper that creates and saves 'niftis_ranked' for the 
        specified dataset, then updates JSON so future runs skip re-ranking.
        """
        paths = self.dataset_paths_dict[dataset_name]
        if 'niftis_ranked' not in paths or not os.path.exists(paths['niftis_ranked']):
            print(f"Initializaing ranked data for: {dataset_name}")
            original_path = paths['niftis']  # The original unranked data
            ranked_path = os.path.splitext(original_path)[0] + '_ranked.npy'

            # Load & rank
            arr = np.load(original_path)
            ranked_arr = self._rank_niftis(arr)
            np.save(ranked_path, ranked_arr)

            # Update the JSON in memory
            self.dataset_paths_dict[dataset_name]['niftis_ranked'] = ranked_path

            # Write back to disk
            with open(self.data_dict_path, 'w') as fw:
                json.dump(self.dataset_paths_dict, fw, indent=4)
    
    def load_regression_dataset(self, dataset_name, nifti_type='niftis', mmap_mode=_DEFAULT_MMAP_MODE):
        paths = self.dataset_paths_dict[dataset_name]
        if mmap_mode is _DEFAULT_MMAP_MODE:
            mmap_mode = self.mmap_mode
        data = {
            'design_matrix': np.load(paths['design_matrix'], mmap_mode=mmap_mode),
            'contrast_matrix': np.load(paths['contrast_matrix'], mmap_mode=mmap_mode),
            'niftis': np.load(paths[nifti_type], mmap_mode=mmap_mode)
        }
        return data
    
    def _read_json(self):
        with open(self.data_dict_path, "r") as f:
            return json.load(f)
        
    def output_info(self):
        payload = self._read_json()["neuroimaging_regression"]
        return {
            "output_ftype": payload.get("output_ftype", "nifti"),
            "mask_path": payload.get("mask_path", None)
        }
