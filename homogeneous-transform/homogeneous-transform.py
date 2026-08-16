import numpy as np

def apply_homogeneous_transform(T, points):
    """
    Apply 4x4 homogeneous transform T to 3D point(s).
    """
    # Your code here
    T = np.array(T)
    points = np.array(points)

    points = points.reshape(-1,3)
    num_points = points.shape[0]
    points_h = np.hstack([points, np.ones((num_points, 1))])

    matrix = (T @ points_h.T).T
    out = matrix[:, :-1]

    if matrix.shape[0] == 1:
        out = out.flatten()

    return out