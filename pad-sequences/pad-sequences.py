import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    L = max_len if max_len else max((len(seq) for seq in seqs), default=0)
    out = np.full((len(seqs), L), pad_value, dtype=int)

    for i, seq in enumerate(seqs):
        out[i, :len(seq)] = seq[:L]

    return out
        
