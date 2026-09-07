import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from regressao_multipla import MRegression
from metricas import *

X1 = np.array(
    [2,8,11,10,8,4,2,2,9,8]
)

X2 = np.array(
    [50,110,120,550,295,200,375,52,100,300]
)

Y = np.array(
    [9.95,24.45,31.75,35,25.02,16.86,14.38,9.6,24.35,27.5]
)

X = np.column_stack(
    (np.ones(X2.shape[0]),X1,X2)
)

X = np.column_stack(
    (X1,X2)
)

modelo = MRegression(X,Y)
modelo.fit()

print(f"valores parametros: {modelo.beta}")
y_pred = modelo.predict(X)
print(f"valores preditos: {y_pred}")
print(f"R^2 = {r2_score(Y, y_pred)}")

fig = go.Figure()
fig.add_scatter3d(x=X1,y=X2,z=Y, mode="markers",marker=dict(color="red", size=5), name="Dados Originais")

fig.add_scatter3d(x=X1,y=X2,z=y_pred, mode="markers",marker=dict(color="green", size=5), name="Dados Previstos")


fig.show()

print(np.linspace(min(X1),max(X1),2))
print(np.linspace(min(X2),max(X2),2))

x1_grind, x2_grind = np.meshgrid(
    np.linspace(min(X1),max(X1),10),
    np.linspace(min(X2),max(X2),10)
)

y_grind = modelo.beta[0] + modelo.beta[1] * x1_grind + modelo.beta[2] * x2_grind 

fig = plt.figure()

ax = fig.add_subplot(projection="3d")
ax.scatter(X1,X2,Y, color="red")
ax.plot_surface(x1_grind, x2_grind, y_grind)
plt.show()

# print(x1_grind)
# print(x2_grind)