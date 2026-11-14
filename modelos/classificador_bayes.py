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

        if self.tipo_densidade == "multivariado":
            self.covariancia = np.zeros(
                (n_classes, n_feautures, n_feautures),
                dtype=np.float64
            )

        for idx,c in enumerate(self.classes):
            X_c = X_train[y_train == c]
            
            self.media[idx,:] = X_c.mean(axis=0)
            
            self.variancia[idx,:] = X_c.var(axis=0) + 1e-9 # Soma com epsilon para evitar operações matematicas com zero

            # self.variancia[idx,:] = X_c.var(axis=0)
            self.prior[idx] = X_c.shape[0] / float(n_samples)

            if self.tipo_densidade == "multivariado":
                self.covariancia[idx] = np.cov(X_c, rowvar=False) + 1e-9 * np.eye(n_feautures)

    def predict(self,X_test):
        y_pred = [self.__predict(x) for x in X_test]

        return np.array(y_pred)

    def __predict(self,x):

        posteriores = []

        for idx, classe in enumerate(self.classes):
            prior = np.log(self.prior[idx])
            
            if self.tipo_densidade == "univariado":
                posterior = np.sum(self.densidade_univariada(idx,x))
            elif self.tipo_densidade == "multivariado":
                posterior = np.sum(self.densidade_multivariada(idx,x))

            posterior += prior
            posteriores.append(posterior)
        
        return self.classes[np.argmax(posteriores)]

    def densidade_univariada(self, idx_classe, x):

        media = self.media[idx_classe]
        variancia = self.variancia[idx_classe]

        return -(1/2) * (
            np.log(2 * np.pi * variancia)
            + ((x - media) ** 2) / variancia
        )
    
    def densidade_multivariada(self, idx_classe, x):
    
        media = self.media[idx_classe]
        covariancia = self.covariancia[idx_classe]
    
        n_features = len(x)
    
        determinante = np.linalg.det(covariancia)
        inversa = np.linalg.inv(covariancia)
    
        diferenca = x - media
    
        expoente = diferenca.T @ inversa @ diferenca
    
        return -(1/2) * (
            n_features * np.log(2 * np.pi)
            + np.log(determinante)
            + expoente
        )