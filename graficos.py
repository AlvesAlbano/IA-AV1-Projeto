import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np


def grafico_real_predito(X, y, y_pred, titulo_grafico: str = "Teste"):

    # ==========================================================
    # MODELO COM 1 VARIÁVEL INDEPENDENTE
    # ==========================================================
    if X.shape[1] == 1:

        x = X[:, 0]

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        # Valores reais
        ax.scatter(
            x,
            y,
            color="blue",
            alpha=0.7,
            label="Valores Observados"
        )

        # Valores previstos
        ax.scatter(
            x,
            y_pred,
            color="red",
            alpha=0.7,
            label="Valores Previstos"
        )

        ax.set_title(titulo_grafico)
        ax.set_xlabel("Corrente (mAmp)")
        ax.set_ylabel("Dose de Radiação (rad)")

        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.show()

    # ==========================================================
    # MODELO COM 2 VARIÁVEIS INDEPENDENTES
    # ==========================================================
    elif X.shape[1] == 2:

        x1 = X[:, 0]
        x2 = X[:, 1]

        fig = go.Figure()

        fig.add_scatter3d(
            x=x1,
            y=x2,
            z=y,
            mode="markers",
            marker=dict(color="blue", size=5),
            name="Dados Originais"
        )

        fig.add_scatter3d(
            x=x1,
            y=x2,
            z=y_pred,
            mode="markers",
            marker=dict(color="red", size=5),
            name="Dados Previstos"
        )

        fig.update_layout(
            title=titulo_grafico,
            scene=dict(
                xaxis_title="Corrente (mAmp)",
                yaxis_title="Tempo de Exposição (Min)",
                zaxis_title="Dose de Radiação (rad)"
            )
        )

        fig.show()

    else:
        raise ValueError(
            "A função suporta apenas 1 ou 2 variáveis independentes."
        )


def grafico_plano_regressao(
    X,
    y,
    modelo,
    titulo_grafico: str = "Teste"
):

    # ==========================================================
    # MODELO COM 1 VARIÁVEL INDEPENDENTE
    # ==========================================================
    if X.shape[1] == 1:

        x = X[:, 0]

        # Cria pontos para desenhar a reta
        x_grid = np.linspace(
            np.min(x),
            np.max(x),
            100
        ).reshape(-1, 1)

        # Predições
        y_grid = modelo.predict(x_grid)

        # Gráfico
        plt.figure(figsize=(10, 6))

        # Dados reais
        plt.scatter(
            x,
            y,
            color="red",
            alpha=0.7,
            label="Valores Observados"
        )

        # Reta de regressão
        plt.plot(
            x_grid[:, 0],
            y_grid,
            linewidth=2
            # label="Reta de Regressão"
        )

        plt.title(titulo_grafico)
        plt.xlabel("Corrente (mAmp)")
        plt.ylabel("Dose de Radiação (rad)")

        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.show()

    # ==========================================================
    # MODELO COM 2 VARIÁVEIS INDEPENDENTES
    # ==========================================================
    elif X.shape[1] == 2:

        x1 = X[:, 0]
        x2 = X[:, 1]

        x1_grid, x2_grid = np.meshgrid(
            np.linspace(np.min(x1), np.max(x1), 10),
            np.linspace(np.min(x2), np.max(x2), 10)
        )

        # Transforma a malha em uma matriz de entrada para o modelo
        X_grid = np.column_stack([
            x1_grid.ravel(),
            x2_grid.ravel()
        ])

        # Predições do modelo
        y_grid = modelo.predict(X_grid)

        # Volta para o formato da malha
        y_grid = y_grid.reshape(x1_grid.shape)

        # Cria o gráfico 3D
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        # Pontos reais
        ax.scatter(
            x1,
            x2,
            y,
            color="red",
            alpha=0.7,
            label="Valores Observados"
        )

        # Plano de regressão
        ax.plot_surface(
            x1_grid,
            x2_grid,
            y_grid,
            alpha=0.4
        )

        ax.set_title(titulo_grafico)
        ax.set_xlabel("mAmp")
        ax.set_ylabel("Tempo De Exposição")
        ax.set_zlabel("Dose De Radiação")

        ax.legend()

        plt.show()

    else:
        raise ValueError(
            "A função suporta apenas 1 ou 2 variáveis independentes."
        )


def grafico_residuo(
    y,
    y_pred,
    titulo_grafico="Gráfico de Resíduos"
):

    residuos = y - y_pred

    plt.figure(figsize=(8, 5))

    plt.scatter(
        y_pred,
        residuos,
        alpha=0.7
    )

    # Linha de referência em zero
    plt.axhline(
        y=0,
        linestyle="-",
        color="black"
    )

    plt.title(titulo_grafico)
    plt.xlabel("Valor Predito")
    plt.ylabel("Resíduo")

    plt.grid(True, alpha=0.3)

    plt.show()