import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """
    Perform Layer Normalization on sequence data.
    
    Args:
        X: Input array of shape (batch_size, seq_len, feature_dim)
        gamma: Scale parameter of shape (1, 1, feature_dim) or (feature_dim,)
        beta: Shift parameter of shape (1, 1, feature_dim) or (feature_dim,)
        epsilon: Small float to avoid division by zero

    Returns:
        Normalized array of same shape as X
    """
    mean = np.mean(X, axis=-1, keepdims=True)  # shape (batch_size, seq_len, 1)
    var = np.var(X, axis=-1, keepdims=True)    # shape (batch_size, seq_len, 1)
    
    X_normalized = (X - mean) / np.sqrt(var + epsilon)
    out = gamma * X_normalized + beta
    
    return out