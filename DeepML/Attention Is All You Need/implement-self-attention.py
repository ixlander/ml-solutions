def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: (seq_len, d_k)
        K: (seq_len, d_k)
        V: (seq_len, d_v)
    
    Returns:
        (seq_len, d_v)
    """
    
    d_k = Q.shape[1]
    
    scores = np.dot(Q, K.T)  # (seq_len, seq_len)
    
    scores = scores / np.sqrt(d_k)
    
    scores_exp = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    attention_weights = scores_exp / np.sum(scores_exp, axis=1, keepdims=True)
    
    output = np.dot(attention_weights, V)  # (seq_len, d_v)
    
    return output
