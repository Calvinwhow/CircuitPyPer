from glob import glob
from nilearn import image
import os
import pandas as pd
from calvin_utils.neuroimaging_utils.nifti_utils.matrix_utilities import import_nifti_to_numpy_array

def generate_unique_column_name(file_path: str, seen_names: set) -> str:
    """Generates a unique column name based on the file path and a set of already seen names."""
    base_name = os.path.basename(file_path)
    if base_name in seen_names:
        name = os.path.join(os.path.basename(os.path.dirname(file_path)), base_name)
    else:
        name = base_name
    return name

def import_matrices_from_df_series(df_series):
    matrix_df = pd.DataFrame({})
    for index, row in df_series.iterrows():
        file_path = row.values[0]  # Assuming the file paths are in the first column of the CSV

        # Ensure the file exists before trying to open it
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
        
        img = image.load_img(file_path)
        data = img.get_fdata()
        name = index
        matrix_df[name] = data.flatten()
    return matrix_df

def import_matrices_from_csv(csv_path: str) -> pd.DataFrame:
    """Reads a CSV file containing paths to NIFTI files, imports the NIFTI files, flattens them, and creates a DataFrame in the specified format"""
    file_paths = pd.read_csv(csv_path)
    matrix_df = pd.DataFrame({})
    seen_names = set()
    
    for index, row in file_paths.iterrows():
        file_path = row.values[0]  # Assuming the file paths are in the first column of the CSV
        data = import_nifti_to_numpy_array(file_path)
        name = generate_unique_column_name(file_path, seen_names)
        matrix_df[name] = data.flatten()
    return matrix_df

def get_subject_id_from_path(file_path, subject_id_index):
    """Splits path into a list by '/', then indexes location"""
    path_parts = file_path.split(os.sep)
    if 0 <= subject_id_index < len(path_parts):
        return path_parts[subject_id_index]
    else:
        return ""  # Return an empty string if the index is out of range
    
def get_subject_id_from_substring(file_path, substring):
    """Assumes subject ID is a folder name and extracts it from the folder names."""
    if substring in file_path:
       return file_path.split(substring)[1].split(os.sep)[0]
    else:
        return ""
        
def import_matrices_from_folder(connectivity_path, file_pattern='', subject_id_index=None, sub_id_str=None, full_path=False):
    glob_path  = os.path.join(connectivity_path, file_pattern)
    print('I will search: ', glob_path)
    globbed = glob(glob_path)
    matrix_df = pd.DataFrame({})
    seen_names = set()
    for file in globbed:
        prelim_name = os.path.basename(file)
        if subject_id_index is not None:
            id = get_subject_id_from_path(file, subject_id_index=subject_id_index)
            name = id + '_' + os.path.basename(file)
        elif sub_id_str is not None:
            id = get_subject_id_from_substring(file, sub_id_str)
            name = id + '_' + os.path.basename(file)
        elif prelim_name in seen_names:
            name = prelim_name
        elif full_path: 
            name = file
        else:
            name = os.path.basename(file)

        seen_names.add(name)
        matrix_df[name] = image.load_img(file).get_fdata().flatten()
    return matrix_df


