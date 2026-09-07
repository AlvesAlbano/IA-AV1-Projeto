from regressao_multipla import MRegression
import numpy as np
from metricas import *
from residuo import tabela_residuos

dados_arsenio = np.loadtxt("data/arsenio_dataset (1).csv",delimiter=",",skiprows=1,usecols=(0,2,3,4,5))

X = dados_arsenio[:,:-1]
y = dados_arsenio[:,-1]
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
tabela_residuos(y,y_predict,nome_csv="./tabela_residuos_arsenio.csv")
print(f"valores preditos para o X original: \n{y_predict}")
print(f"r2: {r2:.4f}")
print(f"r2 ajustado: {r2_ajustado:.4f}")
print(f"rmse: {rmse:.4f}")
print(f"mae: {mae:.4f}")
print(f"mse: {mse:.4f}")

X_predict = np.array([[30, 5, 5, 0.135]])

y_predict = modelo.predict(X_predict)

print("-------- X inédito: idade = 30 | categoria da agua para beber = 5 | categoria da  ́agua para cozinhar = 5 | arsênio na  ́agua = 0,135 --------")
print(f"valor predito para o X inédito: {y_predict}")


"""
modelo alternativo que use apenas a concentração de arsênio na  ́agua como preditor
"""

X_alternativo = X[:,-1:]

# print(X)
# print(X.shape)
# print(X_alternativo)
# print(X_alternativo.shape)
modelo_alternativo = MRegression(X_alternativo,y)
modelo_alternativo.fit()

y_predict = modelo_alternativo.predict(X_alternativo)
r2 = r2_score(y,y_predict)
r2_ajustado = r2_score_ajustado(r2,X_alternativo)
rmse = root_mean_squared_error(y,y_predict)
mae = mean_absolute_error(y,y_predict)

print("--------MODELO ALTERNATIVO--------")
tabela_residuos(y,y_predict,nome_csv="./tabela_residuos_arsenio_alternativo.csv")
print(f"valores preditos para o X sendo apenas arsênio na água como preditor: \n{y_predict}")
print(f"r2: {r2:.4f}")
print(f"r2 ajustado: {r2_ajustado:.4f}")
print(f"rmse: {rmse:.4f}")
print(f"mae: {mae:.4f}")
print(f"mse: {mse:.4f}")

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
tabela_residuos(y,y_predict,nome_csv="./tabela_residuos_arsenio_sem_intercepto.csv")
print(f"valores preditos para o X original: \n{y_predict}")
print(f"r2: {r2:.4f}")
print(f"r2 ajustado: {r2_ajustado:.4f}")
print(f"rmse: {rmse:.4f}")
print(f"mae: {mae:.4f}")
print(f"mse: {mse:.4f}")