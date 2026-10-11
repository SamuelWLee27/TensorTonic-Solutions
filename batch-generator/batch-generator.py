import numpy as np

def batch_generator(X: list, y: list, batch_size: int, seed: int = 42, drop_last: bool = False):
    """
    Returns a generator of (X_batch, y_batch) tuples.
    """
    # Write code here
    if len(y) != len(X):
        assert("X and y lengths don't match")
    X = np.array(X)
    y = np.array(y)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(X))
    rng.shuffle(indices)
    
    sliced = np.split(indices, range(batch_size, len(indices), batch_size))

    for chunk in sliced:
        if drop_last and len(chunk) < batch_size:
            continue
        yield X[chunk], y[chunk]

    
        