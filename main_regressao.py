import time
import numpy as np

from metricas_regressao import *
from regressao_multipla import MRegression
from hould_out import train_test_split

# from fds import *
# from grafico import *

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

    tempo_treino = fim - inicio
    print(f"predição durou {tempo_treino:.2f}s")

    r2 = r2_score(y_test,y_pred)
    r2_ajustado = r2_score_ajustado(r2,X_test)

    print("----------- MODELO DE REGRESSÃO -----------")
    print(f"R2 SCORE: {r2:.4f}")
    print(f"R2 SCORE AJUSTADO: {r2_ajustado:.4f}")


if __name__ == "__main__":
    data_csv = np.loadtxt("./data/csv_result-dataset_44990_brazilian_houses.csv",delimiter=",",skiprows=1,usecols=(2,3,4,5,9,13))

    X = data_csv[:,:-1]
    y = data_csv[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X,y)

    print(X[:1])
    print(y[:1])
    print(f"Dimensões do dataset: {data_csv.shape}")
    print(f"Dimensões de X: {X.shape}")
    print(f"Dimensões de y: {y.shape}")

    regressao_linear_multipla(X_train, X_test, y_train, y_test)