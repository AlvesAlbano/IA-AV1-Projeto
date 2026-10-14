import os

import numpy as np
from tabela import tabela_informatica_final_regressao,tabela_informatica_final_classificacao

def medidas_metricas_regressao(dados_csv):
    r2 = dados_csv[:,0]
    r2_ajustado = dados_csv[:,1]
    tempo_treino = dados_csv[:,-2]
    tempo_teste = dados_csv[:,-1]
    
    metricas = {
        "R2": r2,
        "R2 Ajustado": r2_ajustado,
        "Tempo Treino (s)": tempo_treino,
        "Tempo Teste (s)": tempo_teste
    }

    resultados = {}

    # print(r2)
    # print(r2_ajustado)
    # print(tempo_treino)
    # print(tempo_teste)

    # print(dados_csv)

    for nome,metrica in metricas.items():
        print(f"-----{nome}-----")
        print(f"media: {metrica.mean():.4f}")
        print(f"desvio padrão: {np.std(metrica,ddof=1):.4f}")

        resultados[nome] = (metrica.mean(), np.std(metrica,ddof=1))

    # print(resultados)
    return resultados

def medidas_metricas_classificacao(dados_csv):

    acuracia = dados_csv[:,0]
    precisao = dados_csv[:,1]
    revocacao = dados_csv[:,2]
    f1_score = dados_csv[:,3]
    tempo_treino = dados_csv[:,-2]
    tempo_teste = dados_csv[:,-1]
    
    metricas = {
        "Acurácia": acuracia,
        "Precisão": precisao,
        "Revocação": revocacao,
        "F1-Score": f1_score,
        "Tempo Treino (s)": tempo_treino,
        "Tempo Teste (s)": tempo_teste
    }

    resultados = {}

    # print(dados_csv)

    for nome,metrica in metricas.items():
        print(f"-----{nome}-----")
        print(f"media: {metrica.mean():.4f}")
        print(f"desvio padrão: {np.std(metrica,ddof=1):.4f}")

        resultados[nome] = (metrica.mean(), np.std(metrica,ddof=1))

    # print(resultados)
    return resultados

if __name__ == "__main__":
    caminho_base = "./resultados/individuais"

    modelos = {
        "regressão":["Regressão Linear Multipla","KNN Regressão (Distância Euclidiana)","KNN Regressão (Distância Manhattan)"],
        "classificação":["KNN Classificação (Distância Euclidiana)","KNN Classificação (Distância Manhattan)","Naive Baiyes (Univariado)"],
    }

    for pasta in os.listdir(caminho_base):
        caminho_pasta = os.path.join(caminho_base,pasta)

        if os.path.isdir(caminho_pasta):
            for arquivo_csv in os.listdir(caminho_pasta):
                if arquivo_csv.endswith("csv"):

                    caminho_csv = os.path.join(caminho_pasta,arquivo_csv)
                    nome_modelo = arquivo_csv.replace("_resultado.csv","") 
                
                    for categoria, modelos_categoria in modelos.items():
                        if nome_modelo in modelos_categoria:
                            categoria_modelo = categoria

                    if categoria_modelo == "regressão":
                        dados_csv = np.loadtxt(caminho_csv,delimiter=";",skiprows=1,usecols=(1,2,3,4))
                        # print(dados_csv)

                        resultado_medidas_metricas = medidas_metricas_regressao(dados_csv)
                        tabela_informatica_final_regressao(nome_modelo,resultado_medidas_metricas)

                    elif categoria_modelo == "classificação":
                        dados_csv = np.loadtxt(caminho_csv,delimiter=";",skiprows=1,usecols=(1,2,3,4,5,6))

                        resultado_medidas_metricas = medidas_metricas_classificacao(dados_csv)
                        tabela_informatica_final_classificacao(nome_modelo,resultado_medidas_metricas)