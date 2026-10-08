import time
import numpy as np

from modelos.classificador_bayes import NaiveBayes

from tabela import tabela_informativa_classificacao
from metricas.metricas_classificacao import *
from validacao_cruzada import k_fold_estratificado_indices

# def classificacao_multivariada(X_train, X_test, y_train, y_test):
#     modelo = NaiveBayes(tipo_densidade="multivariada")

#     inicio = time.perf_counter()
#     modelo.fit(X_train,y_train)
#     fim = time.perf_counter()

#     tempo_treino = fim - inicio
#     print(f"treino durou {tempo_treino:.2f}s")

    
#     inicio = time.perf_counter()
#     y_pred = modelo.predict(X_test)
#     fim = time.perf_counter()

#     tempo_teste = fim - inicio
#     print(f"predição durou {tempo_teste:.2f}s")

#     classes, matriz_conf = matriz_confusao(y_pred,y_test)

#     print(classes)
#     print(matriz_conf)

#     tabela_informativa_classificacao("Naive Baiyes (Univariado)",matriz_conf,tempo_treino,tempo_teste)


def classificacao_univariada(X_train, X_test, y_train, y_test):
    modelo = NaiveBayes(tipo_densidade="univariado")

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

    tabela_informativa_classificacao("Naive Baiyes (Univariado)",matriz_conf,tempo_treino,tempo_teste)

if __name__ == "__main__":
    data_csv = np.loadtxt("./data/csv_result-dataset_44_spambase.csv",delimiter=",",skiprows=1)

    X = data_csv[:,:-1]
    y = data_csv[:,-1]

    print(X[:1])
    print(y[:1])
    print(f"Dimensões do dataset: {data_csv.shape}")
    print(f"Dimensões de X: {X.shape}")
    print(f"Dimensões de y: {y.shape}")

    idx_fold = k_fold_estratificado_indices(X,y)

    for idx_treino, idx_validacao in idx_fold:
        
        X_train, y_train, = X[idx_treino], y[idx_treino]
        X_test, y_test = X[idx_validacao], y[idx_validacao]

        # classificacao_multivariada(X_train, X_test, y_train, y_test)
        classificacao_univariada(X_train, X_test, y_train, y_test)
    # print("---------------- CLASSIFICAÇÃO BAYESIANA USANDO A DENSIDADE MULTIVARIADA ----------------")

    # print("---------------- CLASSIFICAÇÃO BAYESIANA USANDO A DENSIDADE UNIVARIADA ----------------")