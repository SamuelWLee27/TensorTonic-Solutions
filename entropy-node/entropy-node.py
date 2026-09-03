import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    if len(y) == 0:
        return 0.0

    y = np.array(y)
    
    _, counts = np.unique(y, return_counts=True)

    counts = np.array(counts)

    prob = counts/len(y)

    log_prob = np.zeros_like(prob, dtype=float)
    np.log2(prob, out=log_prob, where=(prob > 0))
    
    h = np.sum(-prob * log_prob)

    return float(h)