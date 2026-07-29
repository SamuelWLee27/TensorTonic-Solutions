import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    x = np.array(x)
    p = np.array(p)
    if np.sum(p) != 1:
        raise ValueError("p != 1")
    return np.sum(x * p)
