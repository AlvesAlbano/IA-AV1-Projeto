import numpy as np 

class MRegression:
    def __init__(self, X, y, sem_intercepto:bool=False): #construtor
        self.X = X
        self.y = y
        self.beta = None #parametros
        self.N = X.shape[0]
        self.sem_intercepto = sem_intercepto

    def fit(self): #treinamento

        if not self.sem_intercepto:
            self.X = np.column_stack((np.ones(self.N), self.X))
        self.beta = np.linalg.pinv(self.X.T @ self.X) @ self.X.T @ self.y
        #peseudo-inversa Moore-Pensore, matrizes nao quadraticas ou singulares
    
    def predict(self,X_new):
        N = X_new.shape[0]

        if not self.sem_intercepto:
            X_new = np.column_stack((np.ones(N), X_new))

        return X_new @ self.beta
