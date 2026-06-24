import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x_arr = np.asarray(x, dtype = np.float64)

    return np.where(
        x_arr >= 0,
        1 / (1 + np.exp(-x_arr)),
        np.exp(x_arr) / (1 + np.exp(x_arr))
    )