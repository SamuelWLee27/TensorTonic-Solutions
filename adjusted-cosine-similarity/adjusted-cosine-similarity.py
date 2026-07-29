import numpy as np

def adjusted_cosine_similarity(ratings_matrix, item_i, item_j):
    """
    Compute adjusted cosine similarity between two items.
    """
    # Write code here
    items_i = []
    items_j = []
    user_means = []
    for ratings in ratings_matrix:
        row = np.asarray(ratings, dtype=float)
        i = ratings[item_i]
        j = ratings[item_j]

        if not (i and j):
            continue

        rated = row[row != 0]

        items_i.append(i)
        items_j.append(j)
        user_means.append(rated.mean())

    items_i = np.array(items_i)
    items_j = np.array(items_j)
    user_means = np.array(user_means)

    if isinstance(items_i, int) and items_i == 0:
        return 0.0
    
    denom = np.linalg.norm(items_i - user_means) * np.linalg.norm(items_j - user_means)
    if not denom.any():
        return 0.0

    dot_product = (items_i - user_means) @ (items_j - user_means)
    return dot_product / denom
    