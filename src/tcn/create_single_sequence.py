import numpy as np

def create_single_sequence(X, seq_len):
    """
    Para inference:
    Devuelve SOLO UNA secuencia (1, seq_len, num_features)
    a partir de las últimas seq_len filas.
    """
    if len(X) < seq_len:
        raise ValueError(f"Se necesitan al menos {seq_len} pasos, pero X tiene {len(X)}")

    seq = X[-seq_len:]   # (seq_len, features)

    return np.expand_dims(seq, axis=0)   # → (1, seq_len, features)
