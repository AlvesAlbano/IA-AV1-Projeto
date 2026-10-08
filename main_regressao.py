import time
import numpy as np

from metricas.metricas_regressao import *
from modelos.regressao_multipla import MRegression
from validacao_cruzada import k_fold_indices
from tabela import tabela_informativa_regressao

# rent_amount, property_tax, fire_insurance columns should be ignored if total is taken as the target feature for the task, as they are part of total.

def regressao_linear_multipla(X_train, X_test, y_train, y_test):
    
    modelo = MRegression(X_train,y_train)

    inicio = time.perf_counter()
    modelo.fit()
    fim = time.perf_counter()

    tempo_treino = fim - inicio
    print(f"treino durou {tempo_treino:.2f}s")

    inicio = time.perf_counter()
    y_pred = modelo.predict(X_test)
    fim = time.perf_counter()

    tempo_teste = fim - inicio
    print(f"predição durou {tempo_teste:.2f}s")

    r2 = r2_score(y_test,y_pred)
    r2_ajustado = r2_score_ajustado(r2,X_test)

    print("----------- MODELO DE REGRESSÃO -----------")
    print(f"R2 SCORE: {r2:.4f}")
    print(f"R2 SCORE AJUSTADO: {r2_ajustado:.4f}")

    tabela_informativa_regressao(X_test,y_test,y_pred,"Regressão Linear Multipla",tempo_treino,tempo_teste)


if __name__ == "__main__":
    data_csv = np.loadtxt("./data/csv_result-dataset_44990_brazilian_houses.csv",delimiter=",",skiprows=1,usecols=(2,3,4,5,9,13))

    X = data_csv[:,:-1]
    y = data_csv[:,-1]

    print(X[:1])
    print(y[:1])
    print(f"Dimensões do dataset: {data_csv.shape}")
    print(f"Dimensões de X: {X.shape}")
    print(f"Dimensões de y: {y.shape}")

    idx_fold = k_fold_indices(X)

    for idx_treino, idx_validacao in idx_fold:
        
        X_train, y_train, = X[idx_treino], y[idx_treino]
        X_test, y_test = X[idx_validacao], y[idx_validacao]

        regressao_linear_multipla(X_train, X_test, y_train, y_test)