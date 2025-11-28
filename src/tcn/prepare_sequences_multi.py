import numpy as np

def create_sequences_multi(X, Y, seq_len, horizons=[1, 12, 24, 72, 168]):
    xs, ys = [], []

    max_h = max(horizons)

    for i in range(len(X) - seq_len - max_h):

        # ventana de entrada
        seq_x = X[i : i + seq_len]

        # vector de 5 targets
        seq_y = []

        for h in horizons:
            seq_y.append(Y[i + seq_len + h])

        xs.append(seq_x)
        ys.append(seq_y)

    xs = np.array(xs)
    ys = np.array(ys)

    return xs, ys
