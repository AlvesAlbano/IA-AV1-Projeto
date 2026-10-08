import numpy as np

def k_fold_indices(X,shuffle:bool=False,random_state:int=42,k=10):
    folds = []

    idx_fold = np.arange(len(X))

    if shuffle:
        embaralho = np.random.default_rng(random_state)
        embaralho.shuffle(idx_fold)

    tamanhos = np.full(k, len(X) // k)
    tamanhos[:len(X) % k] += 1

    inicio = 0

    for tamanho in tamanhos:
        fim = inicio + tamanho

        idx_teste = idx_fold[inicio:fim]
        idx_treino = np.concatenate([
            idx_fold[:inicio],
            idx_fold[fim:]
        ])

        folds.append((idx_treino, idx_teste))

        inicio = fim

    return folds

def k_fold_estratificado_indices(X,y,shuffle:bool=False,random_state:int=42,k:int=10):

    folds = [
        [] for fold in range(k)
    ]

    for classe in np.unique(y):
        idxs = np.where(y == classe)[0]

        if shuffle:
            embaralho = np.random.default_rng(random_state)
            embaralho.shuffle(idxs)

        for i,idx in enumerate(idxs):
            folds[i % k].append(idx)

    todos_idx = np.arange(len(X))

    idx_fold = []

    for i in range(k):
        idx_teste = np.array(folds[i])
        idx_treino = np.setdiff1d(todos_idx,idx_teste)

        idx_fold.append((idx_treino,idx_teste))

    return idx_fold

def medidas_metricas_regressao(csv:str):
    data_csv = np.loadtxt(csv,delimiter=";",skiprows=1,usecols=(1,2,3,4))

    r2 = data_csv[:,0]
    r2_ajustado = data_csv[:,1]
    tempo_treino = data_csv[:,-2]
    tempo_teste = data_csv[:,-1]
    
    metricas = {
        "R2": r2,
        "R2 Ajustado": r2_ajustado,
        "Tempo Treino (s)": tempo_treino,
        "Tempo Teste (s)": tempo_teste
    }

    # print(r2)
    # print(r2_ajustado)
    # print(tempo_treino)
    # print(tempo_teste)

    # print(data_csv)

    for nome,metrica in metricas.items():
        print(f"-----{nome}-----")
        print(f"media: {metrica.mean():.2f}")
        print(f"desvio padrão: {np.std(metrica,ddof=1):.2f}")

def medidas_metricas_classificacao(csv:str):
    data_csv = np.loadtxt(csv,delimiter=";",skiprows=1,usecols=(1,2,3,4,5,6))

    acuracia = data_csv[:,0]
    precisao = data_csv[:,1]
    revocacao = data_csv[:,2]
    f1 = data_csv[:,3]
    tempo_treino = data_csv[:,-2]
    tempo_teste = data_csv[:,-1]
    
    metricas = {
        "Acurácia": acuracia,
        "Precisão": precisao,
        "Revocação": revocacao,
        "F1-Score": f1,
        "Tempo Treino (s)": tempo_treino,
        "Tempo Teste (s)": tempo_teste
    }

    # print(data_csv)

    # print(acuracia)
    # print(precisao)
    # print(revocacao)
    # print(f1)
    # print(tempo_treino)
    # print(tempo_teste)

    for nome,metrica in metricas.items():
        print(f"-----{nome}-----")
        print(f"media: {metrica.mean():.2f}")
        print(f"desvio padrão: {np.std(metrica,ddof=1):.2f}")

if __name__ == "__main__":
    print("------ Regressão Linear Multipla ------")
    medidas_metricas_regressao("./resultados/individuais/Regressão Linear Multipla/Regressão Linear Multipla_resultado.csv")

    print("------ KNN (Distância Euclidiana) ------")
    medidas_metricas_classificacao("./resultados/individuais/KNN (Distância Euclidiana)/KNN (Distância Euclidiana)_resultado.csv")

    print("------ KNN (Distância Manhattan) ------")
    medidas_metricas_classificacao("./resultados/individuais/KNN (Distância Manhattan)/KNN (Distância Manhattan)_resultado.csv")

    print("------ Naive Baiyes (Univariado) ------")
    medidas_metricas_classificacao("./resultados/individuais/Naive Baiyes (Univariado)/Naive Baiyes (Univariado)_resultado.csv")