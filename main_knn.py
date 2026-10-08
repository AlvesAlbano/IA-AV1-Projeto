import time
import numpy as np

from modelos.knn import KNN
from tabela import tabela_informativa_classificacao
from metricas.metricas_classificacao import *
from validacao_cruzada import k_fold_estratificado_indices

# rent_amount, property_tax, fire_insurance columns should be ignored if total is taken as the target feature for the task, as they are part of total.

# 0 = não é spam
# 1 = é spam

def classificacao_euclidiana(X_train, X_test, y_train, y_test,k_valor):

    modelo = KNN(task="classificação",distancia="euclidiana",k=k_valor)
    
    inicio = time.perf_counter()
    modelo.fit(X_train,y_train)
    fim = time.perf_counter()

    tempo_treino = fim - inicio
    print(f"treino durou {tempo_treino:.2f}s")
    
    inicio = time.perf_counter()
    y_pred = modelo.predict(X_test)
    fim = time.perf_counter()

    tempo_teste = fim - inicio
    print(f"predição durou {tempo_teste:.2f}s")

    classes, matriz_conf = matriz_confusao(y_pred,y_test)

    # print(FN)
    print(classes)
    print(matriz_conf)

    tabela_informativa_classificacao("KNN (Distância Euclidiana)",matriz_conf,tempo_treino,tempo_teste)

def classificacao_manhattan(X_train, X_test, y_train, y_test,k_valor):

    modelo = KNN(task="classificação",distancia="manhattan",k=k_valor)
    
    inicio = time.perf_counter()
    modelo.fit(X_train,y_train)
    fim = time.perf_counter()

    tempo_treino = fim - inicio
    print(f"treino durou {tempo_treino:.2f}s")

    inicio = time.perf_counter()
    y_pred = modelo.predict(X_test)
    fim = time.perf_counter()

    tempo_teste = fim - inicio
    print(f"predição durou {tempo_teste:.2f}s")

    classes, matriz_conf = matriz_confusao(y_pred,y_test)

    print(classes)
    print(matriz_conf)

    tabela_informativa_classificacao("KNN (Distância Manhattan)",matriz_conf,tempo_treino,tempo_teste)


if __name__ == "__main__":
    data_csv = np.loadtxt("./data/csv_result-dataset_44_spambase.csv",delimiter=",",skiprows=1)

    X = data_csv[:,:-1]
    y = data_csv[:,-1]

    print(X[:1])
    print(y[:1])
    print(f"Dimensões do dataset: {data_csv.shape}")
    print(f"Dimensões de X: {X.shape}")
    print(f"Dimensões de y: {y.shape}")

    idx_fold = k_fold_estratificado_indices(X,y,shuffle=True)
    k_valor = 5

    # for idx_treino, idx_validacao in idx_fold:
        
    #     X_train, y_train, = X[idx_treino], y[idx_treino]
    #     X_test, y_test = X[idx_validacao], y[idx_validacao]

    #     print("---------------- CLASSIFICAÇÃO USANDO A DISTÂNCIA EUCLIDIANA ----------------")
    #     classificacao_euclidiana(X_train, X_test, y_train, y_test,k_valor)
    #     print("---------------- CLASSIFICAÇÃO USANDO A DISTÂNCIA DE MANHATTAN ----------------")
    #     classificacao_manhattan(X_train, X_test, y_train, y_test,k_valor)
