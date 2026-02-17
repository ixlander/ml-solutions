import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray,
                W_q: np.ndarray,
                W_k: np.ndarray,
                W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V

def self_attention(Q: np.ndarray,
                   K: np.ndarray,
                   V: np.ndarray) -> np.ndarray:
    
    d_k = Q.shape[1]
    
    # (seq_len, seq_len)
    scores = Q @ K.T

    scores = scores / np.sqrt(d_k)
    
    scores = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / np.sum(weights, axis=1, keepdims=True)
    
    # (seq_len, d_k)
    output = weights @ V
    
    return output

def multi_head_attention(Q: np.ndarray,
                         K: np.ndarray,
                         V: np.ndarray,
                         n_heads: int) -> np.ndarray:
    
    seq_len, d_model = Q.shape
    
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
    
    d_head = d_model // n_heads
    
    Q = Q.reshape(seq_len, n_heads, d_head)
    K = K.reshape(seq_len, n_heads, d_head)
    V = V.reshape(seq_len, n_heads, d_head)
    
    heads = []
    
    for h in range(n_heads):
        Q_h = Q[:, h, :]
        K_h = K[:, h, :]
        V_h = V[:, h, :]
        
        head_output = self_attention(Q_h, K_h, V_h)
        heads.append(head_output)
    
    output = np.concatenate(heads, axis=1)
    
    return output
