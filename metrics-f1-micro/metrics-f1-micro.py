import numpy as np

def f1_micro(y_true: list[int], y_pred: list[int]) -> float:
    """
    Returns the micro-averaged F1 score as a Python float rounded to four decimals.
    """
    # Write code here
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    num_classes = max(y_true.max(), y_pred.max()) + 1
    cm = np.bincount(num_classes * y_true + y_pred, minlength=num_classes**2).reshape(num_classes, num_classes)

    total_correct = np.sum(np.diagonal(cm))
    col_sums = np.sum(cm, axis=0)
    row_sums = np.sum(cm, axis=1)

    denominator = np.sum(col_sums + row_sums)

    if denominator == 0:
        return 0.0

    
    score = (2 * total_correct) / denominator
    return round(float(score), 4)