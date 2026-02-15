import numpy as np
import pandas as pd

def convert_ijk_to_mni(ijk, affine):
    return affine @ np.append(ijk, 1)

def convert_mni_to_ijk(mni, affine):
    inv_affine = np.linalg.pinv(affine)
    return inv_affine @ np.append(mni, 1)

def convert_talairach_to_mni(talairach_coords):
    """
    Convert Talairach coordinates to MNI coordinates using a predefined transformation matrix.

    Parameters:
    -----------
    talairach_coords : np.array
        1D array representing the Talairach coordinates (x, y, z).

    Returns:
    --------
    mni_coords : np.array
        1D array representing the MNI coordinates (x, y, z).
    """
    # Predefined transformation matrix from Talairach to MNI
    talairach_to_mni_matrix = np.array([
        [0.9900, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.9688, 0.0460, 0.0000],
        [0.0000, -0.0485, 0.9189, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ])

    # Convert Talairach coordinates to homogeneous coordinates
    if len(talairach_coords) == 3:
        talairach_coords_homogeneous = np.append(talairach_coords, 1)
    mni_coords_homogeneous = talairach_to_mni_matrix @ talairach_coords_homogeneous
    return mni_coords_homogeneous[:3]

def convert_mni_to_talairach_mni(mni_coords):
    """
    Convert MNI coordinates to Talairach coordinates using a predefined transformation matrix.

    Parameters:
    -----------
    mni_coords : np.array
        1D array representing the Talairach coordinates (x, y, z).

    Returns:
    --------
    tal_coords : np.array
        1D array representing the MNI coordinates (x, y, z).
    """
    # Predefined transformation matrix from Talairach to MNI
    talairach_to_mni_matrix = np.array([
        [0.9900, 0.0000, 0.0000, 0.0000],
        [0.0000, 0.9688, 0.0460, 0.0000],
        [0.0000, -0.0485, 0.9189, 0.0000],
        [0.0000, 0.0000, 0.0000, 1.0000]
    ])
    invmx = np.linalg.pinv(talairach_to_mni_matrix)

    # Convert Talairach coordinates to homogeneous coordinates
    if len(mni_coords) == 3:
        mni_coords = np.append(mni_coords, 1)
    tal_coords = invmx @ mni_coords
    return tal_coords[:3]

def convert_coordinates_in_df(df, x_col, y_col, z_col, conversion_type):
    """
    Convert coordinates from one space to another and store the converted coordinates in new columns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the coordinates to be converted.
    x_col : str
        Column name for the x coordinates.
    y_col : str
        Column name for the y coordinates.
    z_col : str
        Column name for the z coordinates.
    conversion_type : str
        Type of conversion to perform. Options are:
        - 'ijk_to_mni'
        - 'mni_to_ijk'
        - 'talairach_to_mni'
        - 'mni_to_talairach'

    Returns:
    --------
    pd.DataFrame
        DataFrame with the converted coordinates in new columns.
    """
    conversion_functions = {
        'ijk_to_mni': convert_ijk_to_mni,
        'mni_to_ijk': convert_mni_to_ijk,
        'talairach_to_mni': tal2mni,
        'mni_to_talairach': mni2tal
    }

    if conversion_type not in conversion_functions:
        raise ValueError(f"Invalid conversion type: {conversion_type}")

    convert_func = conversion_functions[conversion_type]

    # Prepare new column names
    new_x_col = f"x_{conversion_type.split('_')[-1]}"
    new_y_col = f"y_{conversion_type.split('_')[-1]}"
    new_z_col = f"z_{conversion_type.split('_')[-1]}"

    # Perform the conversion
    converted_coords = df.apply(lambda row: convert_func(np.array([row[x_col], row[y_col], row[z_col]])), axis=1)
    df[[new_x_col, new_y_col, new_z_col]] = pd.DataFrame(converted_coords.tolist(), index=df.index)

    return df


#### MNI 2 TAL BASED ON PUBLICATIONS ####
def mni2tal_matrix():
    """
    Defines the transformation matrices for MNI to Talairach conversion.
    Returns:
        M2T (dict): Dictionary containing rotation and zoom matrices.
    """
    M2T = {}

    # Rotation matrix (pitch correction)
    M2T['rotn'] = np.array([
        [1.0000,  0.0000,   0.0000,   0.0000],
        [0.0000,  0.9988,   0.0500,   0.0000],
        [0.0000, -0.0500,   0.9988,   0.0000],
        [0.0000,  0.0000,   0.0000,   1.0000]
    ])

    # Zoom matrix for above AC/PC line
    M2T['upZ'] = np.array([
        [0.9900,  0.0000,   0.0000,   0.0000],
        [0.0000,  0.9700,   0.0000,   0.0000],
        [0.0000,  0.0000,   0.9200,   0.0000],
        [0.0000,  0.0000,   0.0000,   1.0000]
    ])

    # Zoom matrix for below AC/PC line
    M2T['downZ'] = np.array([
        [0.9900,  0.0000,   0.0000,   0.0000],
        [0.0000,  0.9700,   0.0000,   0.0000],
        [0.0000,  0.0000,   0.8400,   0.0000],
        [0.0000,  0.0000,   0.0000,   1.0000]
    ])

    return M2T

def mni2tal(inpoints):
    """
    Converts MNI coordinates to Talairach coordinates.
    
    Args:
        inpoints (numpy.ndarray): Nx3 or 3xN array of MNI coordinates (X, Y, Z).
    
    Returns:
        numpy.ndarray: Converted Talairach coordinates.
    """
    if len(inpoints.shape) == 2 and inpoints.shape[1] == 3:
        inpoints = inpoints.T  # Ensure 3xN format if Nx3 is given
    
    if inpoints.shape[0] != 3:
        raise ValueError("Input must be a Nx3 or 3xN matrix")

    # Load transformation matrices
    M2T = mni2tal_matrix()

    # Add a row of ones for homogeneous coordinates
    inpoints = np.vstack((inpoints, np.ones((1, inpoints.shape[1]))))

    # Identify points above and below the AC/PC line
    below_ac = inpoints[2, :] < 0

    # Apply transformations
    outpoints = np.zeros_like(inpoints)
    outpoints[:, below_ac] = (M2T['rotn'] @ M2T['downZ'] @ inpoints[:, below_ac])
    outpoints[:, ~below_ac] = (M2T['rotn'] @ M2T['upZ'] @ inpoints[:, ~below_ac])

    # Return only the X, Y, Z coordinates
    outpoints = outpoints[:3, :]

    if outpoints.shape[1] == inpoints.shape[1]:  # If Nx3 input, transpose back
        outpoints = outpoints.T

    return outpoints

def tal2mni_matrix():
    """
    Defines the inverted transformation matrices for Talairach to MNI conversion.
    Returns:
        T2M (dict): Dictionary containing the inverse rotation and zoom matrices.
    """
    T2M = {}

    # Inverted rotation matrix
    T2M['rotn_inv'] = np.linalg.inv(np.array([
        [1.0000,  0.0000,   0.0000,   0.0000],
        [0.0000,  0.9988,   0.0500,   0.0000],
        [0.0000, -0.0500,   0.9988,   0.0000],
        [0.0000,  0.0000,   0.0000,   1.0000]
    ]))

    # Inverted zoom matrix for above AC/PC line
    T2M['upZ_inv'] = np.linalg.inv(np.array([
        [0.9900,  0.0000,   0.0000,   0.0000],
        [0.0000,  0.9700,   0.0000,   0.0000],
        [0.0000,  0.0000,   0.9200,   0.0000],
        [0.0000,  0.0000,   0.0000,   1.0000]
    ]))

    # Inverted zoom matrix for below AC/PC line
    T2M['downZ_inv'] = np.linalg.inv(np.array([
        [0.9900,  0.0000,   0.0000,   0.0000],
        [0.0000,  0.9700,   0.0000,   0.0000],
        [0.0000,  0.0000,   0.8400,   0.0000],
        [0.0000,  0.0000,   0.0000,   1.0000]
    ]))

    return T2M

def tal2mni(inpoints):
    """
    Converts Talairach coordinates to MNI coordinates.
    
    Args:
        inpoints (numpy.ndarray): Nx3 or 3xN array of Talairach coordinates (X, Y, Z).
    
    Returns:
        numpy.ndarray: Converted MNI coordinates.
    """
    if inpoints.shape[0] != 3:
        raise ValueError("Input must be a Nx3 or 3xN matrix")
    T2M = tal2mni_matrix()
    
    inpoints = np.hstack((inpoints, 1))
    inpoints = np.reshape(inpoints, (4,1))
    
    # Identify points above and below the AC/PC line
    outpoints = np.zeros_like(inpoints)
    if inpoints[2, :] < 0:
        outpoints[:] = (T2M['downZ_inv'] @ T2M['rotn_inv'] @ inpoints[:])
    else:
        outpoints[:] = (T2M['upZ_inv'] @ T2M['rotn_inv'] @ inpoints[:])
        
    # Return only the X, Y, Z coordinates
    return outpoints[:3].flatten()

