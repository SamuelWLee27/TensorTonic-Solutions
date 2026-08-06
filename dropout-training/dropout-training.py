import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    if rng is None:
        rng = np.random.default_rng()
    dropout_pattern = rng.choice([1/(1-p), 0.0], size=x.shape, p=[1 - p, p])
    output = x * dropout_pattern
    return (output, dropout_pattern)
    