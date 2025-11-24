import numpy as np

def create_sequences_multi(X, Y_multi, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(Y_multi[i+seq_len])
    return np.array(xs), np.array(ys)
