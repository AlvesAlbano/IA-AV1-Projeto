import time
import numpy as np
from classificador_bayes import NaiveBayes
from hould_out import train_test_split

def classificacao_multivariada(X_train, X_test, y_train, y_test):
    modelo = NaiveBayes(tipo_densidade="multivariada")

    inicio = time.perf_counter()
    # modelo.fit(X_train,y_train)
    fim = time.perf_counter()

    tempo_treino = fim - inicio
    print(f"treino durou {tempo_treino:.2f}s")

    
    inicio = time.perf_counter()
    # y_pred = modelo.predict(X_test)
    fim = time.perf_counter()

    tempo_teste = fim - inicio
    print(f"predição durou {tempo_teste:.2f}s")

def classificacao_univariada(X_train, X_test, y_train, y_test):
    modelo = NaiveBayes(tipo_densidade="univariada")

    inicio = time.perf_counter()
    # modelo.fit(X_train,y_train)
    fim = time.perf_counter()

    tempo_treino = fim - inicio
    print(f"treino durou {tempo_treino:.2f}s")

    inicio = time.perf_counter()
    # y_pred = modelo.predict(X_test)
    fim = time.perf_counter()

    tempo_teste = fim - inicio
    print(f"predição durou {tempo_teste:.2f}s")

if __name__ == "__main__":
    data_csv = np.loadtxt("./data/csv_result-dataset_44_spambase.csv",delimiter=",",skiprows=1)

    X = data_csv[:,:-1]
    y = data_csv[:,-1]

    X_train, X_test, y_train, y_test = train_test_split(X,y)

    print(X[:1])
    print(y[:1])
    print(f"Dimensões do dataset: {data_csv.shape}")
    print(f"Dimensões de X: {X.shape}")
    print(f"Dimensões de y: {y.shape}")

    print("---------------- CLASSIFICAÇÃO BAYESIANA USANDO A DENSIDADE MULTIVARIADA ----------------")
    classificacao_multivariada(X_train, X_test, y_train, y_test)

    print("---------------- CLASSIFICAÇÃO BAYESIANA USANDO A DENSIDADE UNIVARIADA ----------------")
    classificacao_univariada(X_train, X_test, y_train, y_test)