import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    result = len(y_true) == len(y_pred)
    if not result:
        raise AttributeError("wrong len")

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    correct_probs = y_pred[np.arange(len(y_true)), y_true]

    loss = np.array([])
    
    for prob in correct_probs:
         loss = np.append(loss, np.log(prob))

    print(loss)
    return -np.mean(loss)