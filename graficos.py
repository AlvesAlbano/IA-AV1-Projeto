import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np

from itertools import combinations

def grafico_plano_regressao(
    X,
    y,
    modelo,
    xlabel:str,
    ylabel:str,
    titulo_grafico: str = "Teste",
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
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

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

def grafico_real_predito(y,y_pred,titulo_grafico="Teste"):

    plt.figure(figsize=(8, 6))

    # Valores reais x previstos
    plt.scatter(
        y_pred,
        y,
        alpha=0.7
    )

    # Linha ideal: Real = Predito
    limite_min = min(
        np.min(y),
        np.min(y_pred)
    )

    limite_max = max(
        np.max(y),
        np.max(y_pred)
    )

    plt.plot(
        [limite_min, limite_max],
        [limite_min, limite_max],
        linestyle="-",
        linewidth=2,
        color="red",
        label="Predição Ideal"
    )

    plt.title(titulo_grafico)

    plt.xlabel(
        "Valores Preditos"
    )

    plt.ylabel(
        "Valores Reais"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.show()

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

"""
exclusivo para o problema do arsenio
"""

def grafico_planos_regressao(
    X,
    y,
    modelo,
    nomes=None,
    titulo_grafico="Planos de Regressão"
):

    X = np.asarray(X)

    # ==========================================================
    # VERIFICAÇÕES
    # ==========================================================

    if X.ndim != 2:
        raise ValueError("X deve ser uma matriz 2D.")

    if X.shape[1] != 4:
        raise ValueError(
            "Esta função foi criada para exatamente 4 variáveis independentes."
        )

    # Nomes das variáveis
    if nomes is None:
        nomes = [
            "Idade",
            "Beber",
            "Cozinhar",
            "Arsênio na água"
        ]

    # ==========================================================
    # VALORES FIXOS
    # ==========================================================
    # Quando duas variáveis estão no gráfico,
    # as outras duas ficam fixadas na média.

    valores_fixos = np.mean(X, axis=0)

    # ==========================================================
    # COMBINAÇÕES DAS 4 VARIÁVEIS
    # ==========================================================

    combinacoes = list(combinations(range(4), 2))

    # ==========================================================
    # CRIA FIGURA COM 6 GRÁFICOS
    # ==========================================================

    fig = plt.figure(figsize=(9, 5))

    for i, (col_x, col_y) in enumerate(combinacoes):

        # ------------------------------------------------------
        # Seleciona as duas variáveis
        # ------------------------------------------------------

        x1 = X[:, col_x]
        x2 = X[:, col_y]

        # ------------------------------------------------------
        # Cria a malha
        # ------------------------------------------------------

        x1_grid, x2_grid = np.meshgrid(
            np.linspace(
                np.min(x1),
                np.max(x1),
                20
            ),
            np.linspace(
                np.min(x2),
                np.max(x2),
                20
            )
        )

        # ------------------------------------------------------
        # Cria matriz de entrada para o modelo
        # ------------------------------------------------------

        X_grid = np.tile(
            valores_fixos,
            (x1_grid.size, 1)
        )

        # Coloca as variáveis do gráfico na malha
        X_grid[:, col_x] = x1_grid.ravel()
        X_grid[:, col_y] = x2_grid.ravel()

        # ------------------------------------------------------
        # Faz as previsões
        # ------------------------------------------------------

        y_grid = modelo.predict(X_grid)

        y_grid = y_grid.reshape(
            x1_grid.shape
        )

        # ------------------------------------------------------
        # Cria subplot 3D
        # ------------------------------------------------------

        ax = fig.add_subplot(
            2,
            3,
            i + 1,
            projection="3d"
        )

        # ------------------------------------------------------
        # Dados reais
        # ------------------------------------------------------

        ax.scatter(
            x1,
            x2,
            y,
            color="red",
            alpha=0.7,
            s=25,
            label="Valores Observados"
        )

        # ------------------------------------------------------
        # Plano de regressão
        # ------------------------------------------------------

        ax.plot_surface(
            x1_grid,
            x2_grid,
            y_grid,
            alpha=0.4
        )

        # ------------------------------------------------------
        # Configuração dos eixos
        # ------------------------------------------------------

        ax.set_xlabel(
            nomes[col_x]
        )

        ax.set_ylabel(
            nomes[col_y]
        )

        ax.set_zlabel(
            "Dose de Radiação"
        )

        ax.set_title(
            f"{nomes[col_x]} x {nomes[col_y]}"
        )

        # ------------------------------------------------------
        # Legenda
        # ------------------------------------------------------

        ax.legend(
            fontsize=8
        )

    # ==========================================================
    # TÍTULO GERAL
    # ==========================================================

    fig.suptitle(
        titulo_grafico,
        fontsize=16
    )

    # Ajusta o espaço entre os gráficos
    plt.tight_layout()

    plt.show()