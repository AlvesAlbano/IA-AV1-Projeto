import csv
import numpy as np
import matplotlib.pyplot as plt

# Métricas nas quais valores menores representam resultados melhores
METRICAS_MENOR_MELHOR = [
    "MSE",
    "RMSE",
    "MAE",
    "Tempo Treino (s)",
    "Tempo Teste (s)"
]

def separar_dados(arquivo_csv):

    modelos = []
    cabecalho = []
    medias = []
    desvios = []

    # Leitura do CSV sem Pandas
    with open(arquivo_csv, "r", encoding="utf-8-sig", newline="") as arquivo:

        leitor = csv.reader(arquivo, delimiter=";")

        cabecalho = [coluna.strip() for coluna in next(leitor)]

        for linha in leitor:

            if not linha:
                continue

            modelos.append(linha[0].strip())

            medias_linha = []
            desvios_linha = []

            for valor in linha[1:]:

                partes = valor.split("+-")

                if len(partes) != 2:
                    raise ValueError(
                        f"Formato inválido no CSV: {valor}"
                    )

                media = float(partes[0].strip())
                desvio = float(partes[1].strip())

                medias_linha.append(media)
                desvios_linha.append(desvio)

            medias.append(medias_linha)
            desvios.append(desvios_linha)

    return (
        modelos,
        cabecalho[1:],
        np.array(medias, dtype=float),
        np.array(desvios, dtype=float)
    )

def normalizar_dados(medias, metricas):

    valores_normalizados = np.full_like(medias, np.nan, dtype=float)

    for i, metrica in enumerate(metricas):

        valores = medias[:, i]

        # Ignorar valores NaN
        validos = valores[~np.isnan(valores)]

        if len(validos) == 0:
            continue

        minimo = np.min(validos)
        maximo = np.max(validos)

        # Evitar divisão por zero
        if maximo == minimo:

            valores_normalizados[:, i] = np.where(
                np.isnan(valores),
                np.nan,
                1.0
            )

        else:

            if metrica in METRICAS_MENOR_MELHOR:

                normalizados = (
                    (maximo - valores) / (maximo - minimo)
                )

            else:

                normalizados = (
                    (valores - minimo) / (maximo - minimo)
                )

            valores_normalizados[:, i] = normalizados

    return valores_normalizados

def criar_grafico_radar(
    csv,
    metricas,
    titulo,
    arquivo
):

    modelos, colunas, medias, desvios = separar_dados(csv)

    print(medias)
    # Identificar as posições das métricas
    indices = []

    for metrica in metricas:

        if metrica not in colunas:
            raise ValueError(
                f"Métrica '{metrica}' não encontrada no CSV."
            )

        indices.append(colunas.index(metrica))

    medias = medias[:, indices]
    desvios = desvios[:, indices]

    # Normalização
    # valores_normalizados = normalizar_dados(medias, metricas)
    valores_normalizados = medias

    n_metricas = len(metricas)
    n_modelos = len(modelos)

    if n_metricas < 3:
        raise ValueError(
            "Um gráfico radar precisa de pelo menos 3 métricas."
        )

    # Calcular ângulos
    angulos = np.linspace(
        0,
        2 * np.pi,
        n_metricas,
        endpoint=False
    )

    angulos_fechados = np.concatenate(
        (angulos, [angulos[0]])
    )

    # Definir quantidade de subgráficos
    n_colunas = 2
    n_linhas = int(np.ceil(n_modelos / n_colunas))

    fig, axs = plt.subplots(
        n_linhas,
        n_colunas,
        figsize=(14, 6 * n_linhas),
        subplot_kw={"polar": True}
    )

    axs = np.array(axs).flatten()

    # Criar um radar para cada modelo
    for i, modelo in enumerate(modelos):

        ax = axs[i]

        valores = valores_normalizados[i, :]

        valores_fechados = np.concatenate(
            (valores, [valores[0]])
        )

        ax.plot(
            angulos_fechados,
            valores_fechados,
            linewidth=2,
            marker="o",
            markersize=5,
            label=modelo
        )

        if np.all(np.isfinite(valores)):
            ax.fill(
                angulos_fechados,
                valores_fechados,
                alpha=0.15
            )

        # Configurar eixos
        ax.set_xticks(angulos)
        ax.set_xticklabels(
            metricas,
            fontsize=9
        )

        ax.set_ylim(0, 1)

        ax.set_yticks([
            0.2,
            0.4,
            0.6,
            0.8,
            1.0
        ])

        ax.set_yticklabels(
            ["0.2", "0.4", "0.6", "0.8", "1.0"],
            fontsize=7
        )

        ax.set_title(
            modelo,
            fontsize=12,
            fontweight="bold",
            pad=20
        )

        ax.grid(
            linestyle="--",
            alpha=0.5
        )

    # Remover subgráficos vazios
    for j in range(n_modelos, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(
        titulo,
        fontsize=17,
        fontweight="bold",
        y=1.02
    )

    plt.tight_layout()

    plt.savefig(
        arquivo,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

# ==========================================
# GRÁFICO DE CLASSIFICAÇÃO
# ==========================================

def grafico_classificacao(csv):

    metricas = [
        "Acurácia",
        "Precisão",
        "Revocação",
        "Especificidade",
        "F1-Score"
    ]

    criar_grafico_radar(
        csv,
        metricas,
        "Comparação dos Modelos de Classificação",
        "comparacao_classificacao_radar.png"
    )

# ==========================================
# GRÁFICO DE REGRESSÃO
# ==========================================

def grafico_regressao(csv):

    metricas = [
        "MSE",
        "RMSE",
        "MAE",
        "R2",
        "R2 Ajustado"
    ]

    criar_grafico_radar(
        csv,
        metricas,
        "Comparação dos Modelos de Regressão",
        "comparacao_regressao_radar.png"
    )

def grafico_barras_tempo(csv, titulo, arquivo):

    modelos, colunas, medias, desvios = separar_dados(csv)

    metricas = [
        "Tempo Treino (s)",
        "Tempo Teste (s)"
    ]

    # Localizar as colunas
    indices = [colunas.index(m) for m in metricas]

    # Selecionar os tempos originais, sem normalização
    tempos = medias[:, indices]
    erros = desvios[:, indices]

    n_modelos = len(modelos)

    x = np.arange(n_modelos)
    largura = 0.35

    fig, ax = plt.subplots(figsize=(14, 7))

    # Barras de treinamento
    ax.bar(
        x - largura / 2,
        tempos[:, 0],
        largura,
        yerr=erros[:, 0],
        capsize=4,
        label="Tempo de Treinamento"
    )

    # Barras de teste
    ax.bar(
        x + largura / 2,
        tempos[:, 1],
        largura,
        yerr=erros[:, 1],
        capsize=4,
        label="Tempo de Teste"
    )

    # Configurações
    ax.set_title(
        titulo,
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("Modelos")
    ax.set_ylabel("Tempo (segundos)")

    ax.set_xticks(x)
    ax.set_xticklabels(
        modelos,
        rotation=25,
        ha="right"
    )

    ax.set_ylim(bottom=0)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    ax.legend(title="Tempos")

    plt.tight_layout()

    plt.savefig(
        arquivo,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

def grafico_tempo_classificacao(csv):

    grafico_barras_tempo(
        csv,
        "Comparação dos Tempos - Classificação",
        "tempo_classificacao.png"
    )

def grafico_tempo_regressao(csv):

    grafico_barras_tempo(
        csv,
        "Comparação dos Tempos - Regressão",
        "tempo_regressao.png"
    )

def grafico_metricas_erro_regressao(csv):

    modelos, colunas, medias, desvios = separar_dados(csv)

    metricas = ["MSE", "RMSE", "MAE"]

    indices = [colunas.index(m) for m in metricas]

    valores = medias[:, indices]
    erros = desvios[:, indices]

    n_modelos = len(modelos)

    # Criar três subgráficos
    fig, axs = plt.subplots(
        1,
        3,
        figsize=(18, 6)
    )

    # Criar um gráfico para cada métrica
    for i, metrica in enumerate(metricas):

        ax = axs[i]

        ax.bar(
            np.arange(n_modelos),
            valores[:, i],
            yerr=erros[:, i],
            capsize=5,
            width=0.6,
            label=metrica
        )

        ax.set_title(
            metrica,
            fontsize=14,
            fontweight="bold"
        )

        ax.set_xlabel("Modelos")
        ax.set_ylabel("Valor")

        ax.set_xticks(np.arange(n_modelos))

        ax.set_xticklabels(
            modelos,
            rotation=45,
            ha="right",
            fontsize=9
        )

        ax.set_ylim(bottom=0)

        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4
        )

    fig.suptitle(
        "Comparação das Métricas de Erro - Regressão",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout()

    plt.savefig(
        "metricas_erro_regressao.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

def grafico_r2_regressao(csv):

    modelos, colunas, medias, desvios = separar_dados(csv)

    metricas = ["R2", "R2 Ajustado"]
    indices = [colunas.index(m) for m in metricas]

    valores_originais = medias[:, indices]
    erros_originais = desvios[:, indices]

    # Substituir valores negativos por zero e limitar valores acima de 1
    valores = np.clip(valores_originais, 0, 1)

    # Garantir desvios padrão não negativos
    erros = np.maximum(erros_originais, 0)

    # Se a média original for negativa, zerar também seu desvio padrão
    erros[valores_originais < 0] = 0

    # Limitar as barras de erro para permanecerem entre 0 e 1
    erros_inferiores = np.minimum(erros, valores)
    erros_superiores = np.minimum(erros, 1 - valores)

    n_modelos = len(modelos)

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    for i, metrica in enumerate(metricas):

        ax = axs[i]

        erros_grafico = np.array([
            erros_inferiores[:, i],
            erros_superiores[:, i]
        ])

        ax.bar(
            np.arange(n_modelos),
            valores[:, i],
            yerr=erros_grafico,
            capsize=5,
            width=0.6,
            label=metrica
        )

        ax.set_title(metrica, fontsize=14, fontweight="bold")
        ax.set_xlabel("Modelos")
        ax.set_ylabel("Coeficiente de Determinação")

        ax.set_xticks(np.arange(n_modelos))
        ax.set_xticklabels(
            modelos,
            rotation=45,
            ha="right",
            fontsize=9
        )

        # Fixar escala entre 0 e 1
        ax.set_ylim(0, 1)

        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Comparação de R² e R² Ajustado - Regressão",
        fontsize=17,
        fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig("r2_regressao.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

# ==========================================
# EXECUÇÃO
# ==========================================

if __name__ == "__main__":

    arquivo_regressao = "tabela_informatica_final_regressão.csv"
    arquivo_classificacao = "tabela_informatica_final_classificação.csv"

    # Classificação
    grafico_classificacao(
        arquivo_classificacao
    )

    grafico_tempo_classificacao(
        arquivo_classificacao
    )

    grafico_metricas_erro_regressao(
        arquivo_regressao
    )

    grafico_r2_regressao(
        arquivo_regressao
    )

    grafico_tempo_regressao(
        arquivo_regressao
    )
