import csv

from pathlib import Path
from metricas.metricas_classificacao import *
from metricas.metricas_regressao import *

def tabela_informativa_regressao(X_test,y_test,y_pred,nome_modelo,tempo_treino,tempo_teste):
    # Cria a pasta automaticamente caso ela não exista
    pasta = Path("./resultados/individuais") / nome_modelo
    pasta.mkdir(parents=True, exist_ok=True)

    # Caminho do arquivo CSV
    nome_csv = pasta / f"{nome_modelo}_resultado.csv"
    arquivo_existe = nome_csv.exists()

    r2 = r2_score(y_test,y_pred)
    r2_ajustado = r2_score_ajustado(r2,X_test)

    with open(nome_csv, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")

        if not arquivo_existe:
            escritor.writerow([
                "Classificador",
                "R2",
                "R2 Ajustado",
                "Tempo Treino (s)",
                "Tempo Teste (s)"
            ])

        escritor.writerow([
            nome_modelo,
            f"{r2:.4f}",
            f"{r2_ajustado:.4f}",
            f"{tempo_treino:.2f}",
            f"{tempo_teste:.2f}"
        ])

def tabela_informativa_classificacao(nome_modelo,matriz_confusao,tempo_treino,tempo_teste):
    
    # Cria a pasta automaticamente caso ela não exista
    pasta = Path("./resultados/individuais") / nome_modelo
    pasta.mkdir(parents=True, exist_ok=True)

    # Caminho do arquivo CSV
    nome_csv = pasta / f"{nome_modelo}_resultado.csv"
    arquivo_existe = nome_csv.exists()

    VP = matriz_confusao[0][0]
    FN = matriz_confusao[0][1]
    FP = matriz_confusao[1][0]
    VN = matriz_confusao[1][1]

    acuracia = accuracy(VP,FN,FP,VN)
    precisao = precision(VP,FP)
    revocacao = recall(VP,FN)
    f1 = f1_score(precisao,revocacao)

    with open(nome_csv, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")

        if not arquivo_existe:
            escritor.writerow([
                "Classificador",
                "Acurácia",
                "Precisão",
                "Revocação",
                "F1-Score",
                "Tempo Treino (s)",
                "Tempo Teste (s)"
            ])

        escritor.writerow([
            nome_modelo,
            f"{acuracia:.4f}",
            f"{precisao:.4f}",
            f"{revocacao:.4f}",
            f"{f1:.4f}",
            f"{tempo_treino:.2f}",
            f"{tempo_teste:.2f}"
        ])

def tabela_informatica_final_regressao(nome_modelo,medidas_metricas):

    # {'R2': (np.float64(0.7546099999999999), np.float64(0.14241095814578314)), 'R2 Ajustado': (np.float64(0.75347), np.float64(0.14309212300705676)), 'Tempo Treino (s)': (np.float64(0.006999999999999999), np.float64(0.01888562063228706)), 'Tempo Teste (s)': (np.float64(0.0), np.float64(0.0))}

    nome_arquivo = "tabela_informatica_final_regressão.csv"
    arquivo_existe = Path(nome_arquivo).exists()

    r2_media,r2_desvio_padrao = medidas_metricas["R2"]
    r2_ajustado_media,r2_ajustado_desvio_padrao = medidas_metricas["R2 Ajustado"]
    tempo_treino_media,tempo_treino_desvio_padrao = medidas_metricas["Tempo Treino (s)"]
    tempo_teste_media,tempo_teste_desvio_padrao = medidas_metricas["Tempo Teste (s)"]

    with open(nome_arquivo, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")

        if not arquivo_existe:
            escritor.writerow([
                "Classificador",
                "R2",
                "R2 Ajustado",
                "Tempo Treino (s)",
                "Tempo Teste (s)"
            ])

        escritor.writerow([
            nome_modelo,
            f"{r2_media:.4f} +- {r2_desvio_padrao:.4f}",
            f"{r2_ajustado_media:.4f} +- {r2_ajustado_desvio_padrao:.4f}",
            f"{tempo_treino_media:.4f} +- {tempo_treino_desvio_padrao:.4f}",
            f"{tempo_teste_media:.4f} +- {tempo_teste_desvio_padrao:.4f}"
        ])

def tabela_informatica_final_classificacao(nome_modelo,medidas_metricas):

    # {'Acurácia': (np.float64(0.8352499999999999), np.float64(0.019716053808452094)), 'Precisão': (np.float64(0.96555), np.float64(0.005986883811949047)), 'Revocação': (np.float64(0.75503), np.float64(0.031256396678653375)), 'F1-Score': (np.float64(0.84712), np.float64(0.02088086843660165)), 'Tempo Treino (s)': (np.float64(0.005), np.float64(0.005270462766947299)), 'Tempo Teste (s)': (np.float64(0.027000000000000003), np.float64(0.008232726023485645))}

    nome_arquivo = "tabela_informatica_final_classificação.csv"
    arquivo_existe = Path(nome_arquivo).exists()

    acuracia_media, acuracia_desvio_padrao = medidas_metricas["Acurácia"]
    precisao_media, precisao_desvio_padrao = medidas_metricas["Precisão"]
    revocacao_media, revocacao_desvio_padrao = medidas_metricas["Revocação"]
    f1_score_media, f1_score_desvio_padrao = medidas_metricas["F1-Score"]

    tempo_treino_media, tempo_treino_desvio_padrao = medidas_metricas["Tempo Treino (s)"]
    tempo_teste_media, tempo_teste_desvio_padrao = medidas_metricas["Tempo Teste (s)"]

    with open(nome_arquivo, "a", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")

        if not arquivo_existe:
            escritor.writerow([
                "Classificador",
                "Acurácia",
                "Precisão",
                "Revocação",
                "F1-Score",
                "Tempo Treino (s)",
                "Tempo Teste (s)"
            ])

        escritor.writerow([
            nome_modelo,
            f"{acuracia_media:.4f} +- {acuracia_desvio_padrao:.4f}",
            f"{precisao_media:.4f} +- {precisao_desvio_padrao:.4f}",
            f"{revocacao_media:.4f} +- {revocacao_desvio_padrao:.4f}",
            f"{f1_score_media:.4f} +- {f1_score_desvio_padrao:.4f}",
            f"{tempo_treino_media:.4f} +- {tempo_treino_desvio_padrao:.4f}",
            f"{tempo_teste_media:.4f} +- {tempo_teste_desvio_padrao:.4f}"
        ])