import numpy as np
from typing import Literal

class NaiveBayes:
    def __init__(self,X_train,y_train,tipo_densidade:Literal["univariada","multivariada"] = "univariada"):
        self.tipo_densidade = tipo_densidade
        self.X_train = X_train
        self.y_train = y_train

    def fit(self):
        n_samples, n_features = self.X_train.shape
        self.classes = np.unique(self.y_train)

        n_classes = len(self.classes)

        self.media = np.zeros(
            (n_classes,n_features)
            ,dtype=np.float64
        )
        
        self.variancia = np.zeros(
            (n_classes,n_features)
            ,dtype=np.float64
        )

        self.covariancia = np.zeros(
            (n_classes,n_features,n_features)
            ,dtype=np.float64
        )

        self.prior = np.zeros(
            n_classes,dtype=np.float64
        )


        for idx,c in enumerate(self.classes):
            X_c = self.X_train[self.y_train == c]

            self.media[idx] = X_c.mean(axis=0)
            self.variancia[idx] = X_c.var(axis=0)
            
            # self.media[idx:] = X_c.mean(axis=0)
            # self.variancia[idx:] = X_c.var(axis=0)
            
            self.prior[idx] = X_c.shape[0] / float(n_samples)

            self.covariancia[idx] = np.cov(X_c,rowvar=False)
        
        self.desvio_padrao = np.sqrt(self.variancia)

    def predict(self):

        previsoes = []

        for x in self.X_train:

            probabilidades = []
            print(x.shape)
            
            if self.tipo_densidade == "univariada":
                densidades = self.densidade_univariada(x)
            elif self.tipo_densidade == "multivariada":
                densidades = self.densidade_multivariada(x)

            for idx, classe in enumerate(self.classes):
                probabilidade = np.prod(densidades[idx])

                posterior = probabilidade * self.prior[idx]
                
                probabilidades.append(posterior)

            previsoes.append(self.classes[np.argmax(probabilidades)])

        return np.array(previsoes)
    
    def densidade_univariada(self,X):

        expoente = (-1/2) * (((X - self.media) / self.desvio_padrao)**2)

        return (1 / np.sqrt(2 * np.pi * self.variancia)) * np.exp(expoente)
    
    def densidade_multivariada(self,X):

        matriz_covariancia = np.cov(X)
        determinante = np.linalg.det(matriz_covariancia)
        inversa = np.linalg.pinv(matriz_covariancia)
        transposta = np.transpose(X - self.media)

        expoente = 0
        
        d = 0

        for idx, media in enumerate(self.media):
            pass


# X_train = np.array([
#     [3,2,1],
#     [4,5,6],
#     [7,8,6],
#     [1,3,2],
#     [5,4,3]
# ])

# y_train = np.array(
#     ["Sim","Não","Não","Sim","Sim"]
# )

# n_samples, n_features = X_train.shape
# classes = np.unique(y_train)

# n_classes = len(classes)

# media = np.zeros((n_classes,n_features),dtype=np.float64)
# variancia = np.zeros((n_classes,n_features),dtype=np.float64)
# prior = np.zeros(n_classes,dtype=np.float64)

# print(f"classes: {classes}")
# print(f"n_classes: {n_classes}")
# print(f"mean: {media}")
# print(f"variancia: {variancia}")
# print(f"prior: {prior}")

# for idx,c in enumerate(classes):

#     print(f"indice: {idx}, classe: {c}")
#     X_c = X_train[y_train == c]
#     media[idx:] = X_c.mean(axis=0)
#     variancia[idx:] = X_c.var(axis=0)
#     prior[idx] = X_c.shape[0] / float(n_samples)

#     print(f"X_c: \n{X_c}")
#     print(f"media: \n{media}")
#     print(f"variancia: \n{variancia}")
#     print(f"prior: {prior}")
