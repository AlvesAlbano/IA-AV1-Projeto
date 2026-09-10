from regressao_multipla import MRegression
import numpy as np
from metricas import *
from residuo import tabela_residuos
from graficos import *

dados_radiacao = np.loadtxt("data/dose_radiacao_expandido.csv",delimiter=",",skiprows=1,usecols=(1,2,3))

X = dados_radiacao[:,1:]
y = dados_radiacao[:,0]

# print(X.shape)
# print(y.shape)

modelo = MRegression(X,y)
modelo.fit()

y_predict = modelo.predict(X)

r2 = r2_score(y,y_predict)
r2_ajustado = r2_score_ajustado(r2,X)
rmse = root_mean_squared_error(y,y_predict)
mae = mean_absolute_error(y,y_predict)
mse = mean_square_error(y,y_predict)

print("--------MODELO BASE--------")
tabela_residuos(y,y_predict,nome_csv="./tabela_residuos_radiacao.csv")
# print(f"valores preditos para o X original: \n{y_predict}")
print(f"r2: {r2:.4f}")
print(f"r2 ajustado: {r2_ajustado:.4f}")
print(f"rmse: {rmse:.4f}")
print(f"mae: {mae:.4f}")
print(f"mse: {mse:.4f}")

print(f"intercepto e coeficientes: {modelo.beta}")
# grafico_real_predito(y,y_predict,titulo_grafico="Grafico de Valor Real x Predito (Radiação)")
# grafico_plano_regressao(X,y,modelo,titulo_grafico="Grafico Do Plano De Regressão (Radiação)")
# grafico_residuo(y,y_predict,titulo_grafico="Grafico De Resíduos (Radiação)")

X_predict = np.array([[15,5.0]])

y_predict = modelo.predict(X_predict)

print("-------- X inédito: mAmp = 15 | Tempo de exposição = 5 --------")
print(f"valor predito para o X inédito: {y_predict}")

"""
intercepto zero
"""

modelo = MRegression(X,y,sem_intercepto=True)
modelo.fit()

y_predict = modelo.predict(X)

r2 = r2_score(y,y_predict)
r2_ajustado = r2_score_ajustado(r2,X)
rmse = root_mean_squared_error(y,y_predict)
mae = mean_absolute_error(y,y_predict)
mse = mean_square_error(y,y_predict)

print("--------MODELO SEM INTERCEPTO--------")
tabela_residuos(y,y_predict,nome_csv="./tabela_residuos_radiacao_sem_intercepto.csv")
# print(f"valores preditos para o X original: \n{y_predict}")
print(f"r2: {r2:.4f}")
print(f"r2 ajustado: {r2_ajustado:.4f}")
print(f"rmse: {rmse:.4f}")
print(f"mae: {mae:.4f}")
print(f"mse: {mse:.4f}")

print(f"coeficientes: {modelo.beta}")
# grafico_real_predito(y,y_predict,titulo_grafico="Grafico de Valor Real x Predito (Radiação - Sem Intercepto)")
# grafico_plano_regressao(X,y,modelo,titulo_grafico="Grafico Do Plano De Regressão (Radiação - Sem Intercepto)")
# grafico_residuo(y,y_predict,titulo_grafico="Grafico De Resíduos (Radiação - Sem Intercepto)")

"""
modelo alternativo que use apenas a corrente como preditor
"""
X_alternativo = X[:,0].reshape(-1,1)

modelo = MRegression(X_alternativo,y)
modelo.fit()

y_predict = modelo.predict(X_alternativo)

r2 = r2_score(y,y_predict)
r2_ajustado = r2_score_ajustado(r2,X_alternativo)
rmse = root_mean_squared_error(y,y_predict)
mae = mean_absolute_error(y,y_predict)
mse = mean_square_error(y,y_predict)

print("--------MODELO ALTERNATIVO--------")
tabela_residuos(y,y_predict,nome_csv="./tabela_residuos_radiacao_alternativo.csv")
# print(f"valores preditos para o X original: \n{y_predict}")
print(f"r2: {r2:.4f}")
print(f"r2 ajustado: {r2_ajustado:.4f}")
print(f"rmse: {rmse:.4f}")
print(f"mae: {mae:.4f}")
print(f"mse: {mse:.4f}")

print(f"intercepto e coeficientes: {modelo.beta}")
# grafico_real_predito(y,y_predict,titulo_grafico="Grafico de Valor Real x Predito (Radiação - Modelo Alternativo)")
# grafico_plano_regressao(X_alternativo,y,modelo,titulo_grafico="Grafico Do Plano De Regressão (Radiação - Modelo Alternativo)")
# grafico_residuo(y,y_predict,titulo_grafico="Grafico De Resíduos (Radiação - Modelo Alternativo)")