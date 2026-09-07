import numpy as np

def r2_score(y_true, y_pred) -> float:
    numerador = np.sum((y_true - y_pred)**2)
    denominador = np.sum((y_true-np.mean(y_true))**2)
    r_score = 1 - (numerador/denominador)
    return r_score

def r2_score_ajustado(r2_score,X):
    # X = (21,4)
    n = X.shape[0]
    p = X.shape[1]

    return 1 - (1 - r2_score) * ((n - 1) / (n - p - 1))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(
        np.mean((y_true - y_pred) **2)
    )

def mean_absolute_error(y_true, y_pred):
    return np.mean(
        np.abs(y_true - y_pred)
    )

def mean_square_error(y_true, y_pred):
    return np.mean((y_true - y_pred) **2)