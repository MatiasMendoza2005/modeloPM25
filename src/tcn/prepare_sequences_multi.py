import numpy as np

def create_sequences_multi(X, Y, seq_len):
    xs, ys = [], []

    for i in range(len(X) - seq_len):
        seq_x = X[i : i + seq_len]
        seq_y = Y[i + seq_len]        # vector de 5 targets ya alineados

        xs.append(seq_x)
        ys.append(seq_y)

    xs = np.array(xs)
    ys = np.array(ys)

    return xs, ys
