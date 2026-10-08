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

def tabela_informativa_final():
    pass