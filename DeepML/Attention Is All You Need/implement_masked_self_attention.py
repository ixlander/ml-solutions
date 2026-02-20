import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Compute scaled masked self-attention.
    
    Q, K, V: shape (seq_len, d_k)
    mask: shape (seq_len, seq_len)
    """
    d_k = Q.shape[-1]
    
    scores = np.dot(Q, K.T) / np.sqrt(d_k)
    
    scores = scores + mask
    
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    output = np.dot(attention_weights, V)
    
    return output