import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    A = np.asarray(A)
    rows = A.shape[0]
    columns = A.shape[1]
    B = np.zeros((columns, rows))
    for row_idx in range(rows):
        for column_idx in range(columns):
            B[column_idx][row_idx] = A[row_idx][column_idx]
    return B
