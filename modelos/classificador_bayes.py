import numpy as np
from typing import Literal

class NaiveBayes:
    def __init__(self,tipo_densidade:Literal["univariado","multivariado"]):
        self.tipo_densidade = tipo_densidade

    def fit(self,X_train,y_train):
        n_samples, n_feautures = X_train.shape
        self.classes = np.unique(y_train)

        n_classes = len(self.classes)
        self.media = np.zeros((n_classes,n_feautures),dtype=np.float64)
        self.variancia = np.zeros((n_classes,n_feautures),dtype=np.float64)
        self.prior = np.zeros(n_classes,dtype=np.float64)

        for idx,c in enumerate(self.classes):
            X_c = X_train[y_train == c]
            self.media[idx,:] = X_c.mean(axis=0)
            self.variancia[idx,:] = X_c.var(axis=0)
            self.prior[idx] = X_c.shape[0] / float(n_samples)

    def predict(self,X_test):
        y_pred = [self.__predict(x) for x in X_test]

        return np.array(y_pred)

    def __predict(self,x):

        if self.tipo_densidade == "univariado":
            posteriores = []

            for idx, classe in enumerate(self.classes):

                prior = np.log(self.prior[idx])

                # posterior = np.sum(
                #     np.log(self.densidade_univariada(idx,x))
                # )
                
                posterior = np.sum(self.densidade_univariada(idx,x))

                posterior += prior

                posteriores.append(posterior)

            return self.classes[np.argmax(posteriores)]
        
        elif self.tipo_densidade == "multivariado":
            return 0

    # def densidade_univariada(self, idx_classe, x):

    #     media = self.media[idx_classe]
    #     variancia = self.variancia[idx_classe]

    #     numerador = np.exp(-((x - media)**2)/(2 * variancia))
    #     denominador = np.sqrt(2 * np.pi * variancia)

    #     return numerador / denominador

    def densidade_univariada(self, idx_classe, x):

        media = self.media[idx_classe]
        variancia = self.variancia[idx_classe]

        # Evita variância exatamente igual a zero
        variancia = np.maximum(variancia, 1e-9)

        return -(1/2) * (
            np.log(2 * np.pi * variancia)
            + ((x - media) ** 2) / variancia
        )

    # def densidade_multivariada(self,x):
    #     matriz_covariancia = np.cov(x)

    #     determinante = np.linalg.det(matriz_covariancia)
    #     inversa = np.linalg.pinv(matriz_covariancia)
    #     transposta = (x - self.media).T
    #     d = x.shape[1]

    #     numerador = 1
    #     denominador = ((2 * np.pi) ** (d/2)) * (determinante ** (1/2))
    #     expoente = np.exp(-(1/2) * transposta * inversa * (x - self.media))
        
    #     return (numerador / denominador) * expoente