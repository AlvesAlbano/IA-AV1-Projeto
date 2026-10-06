import math
import numpy as np

def train_test_split(X,y,teste_size:float=0.3,random_state:int=42):

    if random_state is not None:
        np.random.seed(random_state)

    if len(X) != len(y):
        raise ValueError("deu ruim ai")

    n_samples = len(X)

    print(f"Amostra quantidade: {n_samples}")

    indice = np.random.permutation(n_samples)
    print(f"Indices embaralhados {indice}")

    n_teste = math.ceil(n_samples * teste_size)
    print(f"tamanho da amostra de teste {n_teste}")

    test_indice = indice[:n_teste]
    train_indice = indice[n_teste:]

    print(f"indices de teste: {test_indice}")
    print(f"indice de treino {train_indice}")

    if X.ndim == 1:
        X_train, X_test = X[train_indice], X[test_indice]
    else:
        X_train, X_test = X[train_indice,:], X[test_indice,:]
    
    y_train, y_test = y[train_indice], y[test_indice]
    
    return X_train, X_test, y_train, y_test

def k_fold(k:int=5):
    pass

# X_train, X_test, y_train, y_test = train_test_split(X,y)
# print(f"Dados de treino X_train: {X_train}")
# print(f"Dados de treino y_train: {y_train}")
# print(f"Dados de treino y_test: {X_test}")
# print(f"Dados de treino y_test: {y_test}")