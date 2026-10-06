"""VBM data loading."""

import os
from calvin_utils.file_utils.import_matrices import import_matrices_from_folder 

def import_dataframes_by_tissue(base_directory, shared_glob_path, tissue_to_import, subject_id_index=None, sub_id_str=None):
    """
    Imports dataframes based on tissue types from specified directories and glob name patterns.
    
    Parameters:
    - base_directory (str): The base directory where the data resides.
    - shared_glob_path (str): The shared directory path for all tissues.
    - tissue_to_import (list): List of tissues to be imported.
    - subject_id_index (int): Integer location of the folder name to use for the subject name. Folder integer is based on separators in the file path.
    - sub_id_str (str): The substring that isused in the subject name in the folder path. Will split this out and retain everything from it to the folder separator.
    
    Returns:
    - dict: A dictionary containing dataframes for each specified tissue.
    """

    segments_dict = {}
    for tissue in tissue_to_import:
        glob_path = os.path.join(shared_glob_path, ('*'+tissue+'*'))
        segments_dict[tissue] = glob_path

    dataframes_dict = {}
    nan_handler = {'nan': 0, 'posinf':20, 'neginf':-20}
    for k, v in segments_dict.items():
        dataframes_dict[k] = import_matrices_from_folder(connectivity_path=base_directory, file_pattern=v, convert_nan_to_num=nan_handler, subject_id_index=subject_id_index, sub_id_str=sub_id_str)
        print(f'Imported data {k} data with {dataframes_dict[k].shape[0]} voxels and {dataframes_dict[k].shape[1]} patients')
        print(f'Example filename per subject: {dataframes_dict[k].columns[0]}')
        print('\n--------------------------------\n')

    return dataframes_dict

def _generate_report(tissue, dfs):
    try:
        print(f'Imported {tissue}: {dfs[tissue].shape[0]} voxels, {dfs[tissue].shape[1]} patients')
        print(f'Example subject filename: {dfs[tissue].columns[-1]}')
    except IndexError:
        print(f"Error: Index out of range when accessing columns for {tissue}. The provided path was likely incorrect.")
    if dfs[tissue].shape[1] <= 2:
        print("It is likely you have identically named files. Please provide specify {subject_id_index=int, sub_id_str=str} to extract subject names from the file paths. \
            \n\tPass to {import_segments}: \
            \n\tsubject_id_index (int): Integer location of the folder name to use for the subject name. Folder integer is based on separators in the file path. \
            \n\tsub_id_str (str): The substring that isused in the subject name in the folder path. Will split this out and retain everything from it to the folder separator.") 
    print('--------------------------------')

def import_segments(base_dir, gm_pattern, wm_pattern, csf_pattern, sub_id_index=None, sub_id_str=None, full_path=False) -> dict:
    """
    Import control dataframes for grey matter, white matter, and CSF using specified glob patterns.

    Args:
        base_dir (str): Base directory containing the data.
        gm_pattern (str): Glob pattern for grey matter files.
        wm_pattern (str): Glob pattern for white matter files.
        csf_pattern (str): Glob pattern for CSF files.
        subject_id_index (int): Integer location of the folder name to use for the subject name. Folder integer is based on separators in the file path.
        sub_id_str (str): The substring that isused in the subject name in the folder path. Will split this out and retain everything from it to the folder separator.

    Returns:
        dict: Dictionary with keys 'grey_matter', 'white_matter', 'cerebrospinal_fluid' and corresponding dataframes as values.
    """
    patterns = {'grey_matter': gm_pattern, 'white_matter': wm_pattern, 'cerebrospinal_fluid': csf_pattern}
    dfs = {}
    for tissue, pattern in patterns.items():
        dfs[tissue] = import_matrices_from_folder(connectivity_path=base_dir, file_pattern=pattern, subject_id_index=sub_id_index, sub_id_str=sub_id_str, full_path=full_path)
        _generate_report(tissue, dfs)
    return dfs
