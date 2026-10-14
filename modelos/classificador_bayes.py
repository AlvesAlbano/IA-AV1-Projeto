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

        return -(1/2) * (
            np.log(2 * np.pi * variancia)
            + ((x - media) ** 2) / variancia
        )

    def densidade_multivariada(self,x):
        return 0

class Naive:
    def fit(self, X_train, y_train):
      n_samples, n_features  = X_train.shape
      self._classes = np.unique(y_train)
      n_classes = len(self._classes)
      self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
      self._var = np.zeros((n_classes, n_features), dtype=np.float64)
      self._priors = np.zeros(n_classes, dtype=np.float64)
      for idx, c in enumerate(self._classes): #itera sobre cada classe
         X_c = X_train[y_train == c]
         self._mean[idx,:] = X_c.mean(axis=0) #calcula a media por coluna
         self._var[idx,:] = X_c.var(axis=0) #calcula a variancia por coluna
         self._priors[idx] = X_c.shape[0]/float(n_samples)
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
              prior = np.log(self._priors[idx])
              posterior = np.sum(np.log(self._pdf(idx, x)))
            #   posterior = np.sum(self._log_pdf(idx, x))
              #o log transforma essa multiplicaçao em soma
              #transforma em soma para tornar o calculo numericamente mais estavel
              #o valor pode ficar pequeno, podendo chegar ao ponto do computador representar como zero
              #underflow
              posterior = prior+posterior
              posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]
    def predict(self,X_test):
         y_pred = [self._predict(x) for x in X_test]
         return np.array(y_pred)
    def _pdf(self, class_idx, x): #funcao de densidade
         #class_idx(indice da classe)
         mean = self._mean[class_idx]
         var = self._var[class_idx]
         numerator = np.exp(-((x-mean)**2)/(2*var))
         denominator = np.sqrt(2*np.pi*var)
         return numerator/denominator

    def _log_pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]

        return (
            -0.5 * np.log(2 * np.pi * var)
            - ((x - mean) ** 2) / (2 * var)
        )