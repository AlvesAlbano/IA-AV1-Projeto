import time
import numpy as np

from metricas.metricas_regressao import *
from modelos.regressao_multipla import MRegression
from modelos.knn import KNN
from redes_neurais.mlp import MultilayerPerceptron
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

    mse = mean_square_error(y_test,y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    r2_ajustado = r2_score_ajustado(r2,X_test)

    print("----------- MODELO DE REGRESSÃO -----------")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 SCORE: {r2:.4f}")
    print(f"R2 SCORE AJUSTADO: {r2_ajustado:.4f}")

    tabela_informativa_regressao(X_test,y_test,y_pred,"Regressão Linear Multipla",tempo_treino,tempo_teste)

def regressao_euclidiana(X_train, X_test, y_train, y_test,k_valor):

    modelo = KNN(task="regressão",distancia="euclidiana",k=k_valor)
    
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

    mse = mean_square_error(y_test,y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    r2_ajustado = r2_score_ajustado(r2,X_test)

    print("----------- KNN Regressão (Distância Euclidiana) -----------")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 SCORE: {r2:.4f}")
    print(f"R2 SCORE AJUSTADO: {r2_ajustado:.4f}")

    tabela_informativa_regressao(X_test,y_test,y_pred,"KNN Regressão (Distância Euclidiana)",tempo_treino,tempo_teste)

def regressao_manhattan(X_train, X_test, y_train, y_test,k_valor):

    modelo = KNN(task="regressão",distancia="manhattan",k=k_valor)
    
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

    mse = mean_square_error(y_test,y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    r2_ajustado = r2_score_ajustado(r2,X_test)

    print("----------- KNN Regressão (Distância Manhattan) -----------")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 SCORE: {r2:.4f}")
    print(f"R2 SCORE AJUSTADO: {r2_ajustado:.4f}")

    tabela_informativa_regressao(X_test,y_test,y_pred,"KNN Regressão (Distância Manhattan)",tempo_treino,tempo_teste)

def regressao_mlp_relu(X_train, X_test, y_train, y_test):
    print(f"shape: {X_train.shape[1]} <---------------------------")
    modelo = MultilayerPerceptron(
        X_train.shape[1],
        [3,3,3,3],
        1,
        epochs=125,
        learning_rate=0.1,
        activation_func="relu"
    )
    
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

    mse = mean_square_error(y_test,y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    r2_ajustado = r2_score_ajustado(r2,X_test)

    print("----------- MLP - reLu -----------")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 SCORE: {r2:.4f}")
    print(f"R2 SCORE AJUSTADO: {r2_ajustado:.4f}")

    tabela_informativa_regressao(X_test,y_test,y_pred,"MLP - reLu",tempo_treino,tempo_teste)

def regressao_mlp_sigmoid(X_train, X_test, y_train, y_test):

    modelo = MultilayerPerceptron(
        X_train.shape[1],
        [5,5,5],
        1,
        epochs=125,
        learning_rate=0.1,
        activation_func="sigmoid"
    )
    
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

    mse = mean_square_error(y_test,y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    r2_ajustado = r2_score_ajustado(r2,X_test)

    print("----------- MLP - sigmoid -----------")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 SCORE: {r2:.4f}")
    print(f"R2 SCORE AJUSTADO: {r2_ajustado:.4f}")

    tabela_informativa_regressao(X_test,y_test,y_pred,"MLP - sigmoid",tempo_treino,tempo_teste)

def regressao_mlp_tanh(X_train, X_test, y_train, y_test):

    modelo = MultilayerPerceptron(
        X_train.shape[1],
        [5,5,5],
        1,
        epochs=125,
        learning_rate=0.1,
        activation_func="tanh"
    )
    
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

    mse = mean_square_error(y_test,y_pred)
    rmse = root_mean_squared_error(y_test,y_pred)
    mae = mean_absolute_error(y_test,y_pred)
    r2 = r2_score(y_test,y_pred)
    r2_ajustado = r2_score_ajustado(r2,X_test)

    print("----------- MLP - tanh -----------")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 SCORE: {r2:.4f}")
    print(f"R2 SCORE AJUSTADO: {r2_ajustado:.4f}")

    tabela_informativa_regressao(X_test,y_test,y_pred,"MLP - tanh",tempo_treino,tempo_teste)

def kfold_regressao(X,y):
    k_valor = 15

    idx_fold = k_fold_indices(X)

    for idx_treino, idx_validacao in idx_fold:
        
        X_train, y_train, = X[idx_treino], y[idx_treino]
        X_test, y_test = X[idx_validacao], y[idx_validacao]

        # print(X_test.shape, y_test.shape)
        print("---------------- REGRESSÃO USANDO A DISTÂNCIA EUCLIDIANA ----------------")
        regressao_euclidiana(X_train, X_test, y_train, y_test,k_valor)
        
        print("---------------- REGRESSÃO USANDO A DISTÂNCIA DE MANHATTAN ----------------")
        regressao_manhattan(X_train, X_test, y_train, y_test,k_valor)

        print("---------------- REGRESSÃO LINEAR MULTIPLA ----------------")
        regressao_linear_multipla(X_train, X_test, y_train, y_test)

        print("---------------- REGRESSÃO MLP SIGMOID ----------------")
        regressao_mlp_sigmoid(X_train, X_test, y_train, y_test)
        
        print("---------------- REGRESSÃO MLP RELU ----------------")
        regressao_mlp_relu(X_train, X_test, y_train, y_test)

        print("---------------- REGRESSÃO RELU TANH ----------------")
        regressao_mlp_tanh(X_train, X_test, y_train, y_test)

def main():
    data_csv = np.loadtxt("./data/csv_result-dataset_44990_brazilian_houses.csv",delimiter=",",skiprows=1,usecols=(2,3,4,5,9,13))

    X = data_csv[:,:-1]
    y = data_csv[:,-1]

    print(X[:1])
    print(y[:1])
    print(f"Dimensões do dataset: {data_csv.shape}")
    print(f"Dimensões de X: {X.shape}")
    print(f"Dimensões de y: {y.shape}")

    kfold_regressao(X,y)