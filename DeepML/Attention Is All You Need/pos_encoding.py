import numpy as np

def pos_encoding(position: int, d_model: int):
    """
    Calculate Positional Encoding for Transformers.
    
    Args:
        position: Sequence length (number of positions)
        d_model: Model dimension

    Returns:
        Numpy array of shape (position, d_model) with float16,
        or -1 if position is 0 or d_model <= 0
    """
    if position == 0 or d_model <= 0:
        return -1

    pos_enc = np.zeros((position, d_model), dtype=np.float16)

    for pos in range(position):
        for i in range(0, d_model, 2):
            angle = pos / np.power(10000, (2 * i) / d_model)
            pos_enc[pos, i] = np.sin(angle)
            if i + 1 < d_model:
                pos_enc[pos, i + 1] = np.cos(angle)
    
    return pos_enc