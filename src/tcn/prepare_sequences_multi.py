import numpy as np

def create_sequences_multi(X, Y, seq_len):
    xs, ys = [], []

    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(Y[i+seq_len])   # vector de 5 targets YA alineados

    return np.array(xs), np.array(ys)

