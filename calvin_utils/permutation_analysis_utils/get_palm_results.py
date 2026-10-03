import os
from glob import glob
import pandas as pd
import numpy as np

def aggregate_permutation_p_values(
    unique_file_identifier: str,
    num_perms: int,
    in_dir: str,
    out_dir: str,
) -> pd.DataFrame:
    """
    This function aggregates p-values from multiple permutation files and saves the results as a CSV file.
    
    :param unique_file_identifier: str, unique identifier for the files to be processed
    :param num_perms: int, number of permutations in each file
    :param in_dir: str, input directory where the CSV files are located
    :param out_dir: str, output directory where the results will be saved
    :return: The aggregated p-value matrix.
    """
    if num_perms <= 0:
        raise ValueError("num_perms must be greater than zero")

    # Add files together
    globbed = sorted(glob(os.path.join(in_dir, f'*{unique_file_identifier}*.csv')))
    if not globbed:
        raise FileNotFoundError(
            f"No CSV files containing {unique_file_identifier!r} were found in {in_dir!r}"
        )

    unstarted = True
    for file in globbed:
        if unstarted:
            p_values = pd.read_csv(file, header=None, index_col=False)
            unstarted = False
        else:
            new_p = pd.read_csv(file, header=None, index_col=False)
            p_values = p_values + new_p
            
    # Divide by number of permutations
    p_values = p_values / (len(globbed) * num_perms)

    print(p_values)
    print('Num permutations:', (len(globbed) * num_perms))
    print(np.min(p_values))

    os.makedirs(out_dir, exist_ok=True)
    p_values.to_csv(os.path.join(out_dir, f'{len(globbed) * num_perms}_summed_{unique_file_identifier}p_values.csv'), header=True, index=False)
    return p_values


def main() -> None:
    """Collect interactive inputs and aggregate PALM result files."""
    num_perms = int(input('How many permutations are in each file?'))
    in_dir = str(input('Enter the absolute path to the stored files: '))
    unique_file_identifier = str(input('Enter the unique file identifier common between all files of interest: '))
    out_dir = str(input('Enter the absolute path to the directory you would like to save results to: '))

    aggregate_permutation_p_values(unique_file_identifier, num_perms, in_dir, out_dir)


if __name__ == "__main__":
    main()
