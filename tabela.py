import numpy as np
import csv

from pathlib import Path
from metricas_classificacao import *
from metricas_regressao import *

def tabela_informativa_regressao(y_true, y_pred, nome_csv:str="./tabela_informativa.csv"):
    residuos = y_true - y_pred
    observacao = np.arange(1, len(y_true)+1)

    tabela = np.column_stack(
        [observacao,
        y_true,
        y_pred,
        residuos]
    )

    np.savetxt(
        nome_csv,
        tabela,
        delimiter=";",
        fmt=["%d","%.4f","%.4f","%.4f"],
        header="observacao;valor_observado;valor_ajustado;residuo",
        comments=""
    )
    print("tabela de residuos criada")

def tabela_informativa_classificacao(nome_modelo,matriz_confusao,tempo_treino,tempo_teste):
    
    # Cria a pasta automaticamente caso ela não exista
    pasta = Path("./resultados/individuais") / nome_modelo
    pasta.mkdir(parents=True, exist_ok=True)

    # Caminho do arquivo CSV
    nome_csv = pasta / f"{nome_modelo}_resultado.csv"

    VP = matriz_confusao[0][0]
    FN = matriz_confusao[0][1]
    FP = matriz_confusao[1][0]
    VN = matriz_confusao[1][1]

    acuracia = accuracy(VP,FN,FP,VN)
    precisao = precision(VP,FP)
    revocacao = recall(VP,FN)
    f1 = f1_score(precisao,revocacao)

    with open(nome_csv, "w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo, delimiter=";")

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
