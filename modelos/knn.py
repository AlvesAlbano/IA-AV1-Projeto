import numpy as np
from typing import Literal

class KNN:
    def __init__(self, k=5,task:Literal["classificação","regressão"] = "classificação",distancia:Literal["manhattan","euclidiana"] = "euclidiana"):
        self.k = k
        self.task = task
        self.distancia = distancia

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def distancia_eclidiana(self, x1, x2):
        return np.sqrt(
            np.sum(
                (x1-x2)**2
            )
        )  

    def distancia_manhattan(self, x1, x2):
        return np.sum(
            np.abs(x1-x2)
        )

    def calculate_predictions(self, x):
        distances = None

        if self.distancia == "euclidiana":
            distances = [self.distancia_eclidiana(x, x_train) for x_train in self.X_train]
        elif self.distancia == "manhattan":
            distances = [self.distancia_manhattan(x, x_train) for x_train in self.X_train]
        
        k_indices = np.argsort(distances)[:self.k]    
        
        k_nearest_label = [self.y_train[i] for i in k_indices]   
        
        if self.task == "classificação":
            unique, counts = np.unique(k_nearest_label, return_counts=True)
            
            return unique[np.argmax(counts)]
        elif self.task == "regressão":
            return np.mean(k_nearest_label)

    def predict(self, X_test):
        predictions = [self.calculate_predictions(x) for x in X_test]
        return predictions

# """X_train = [[1,2],
#            [2,3],
#            [3,3],
#            [10,10]]
# y_train = [0,0,1,1] #classificacao binaria
# x_test = [[2.5, 2.5],
#           [10, 10]]
# def calculate_distance(x1, x2): #euclidiana 
#         x1 = np.array(x1) 
#         x2 = np.array(x2)
#         return np.sqrt(np.sum((x1-x2)**2))
# distances = [calculate_distance(x_test, x_train)
#                      for x_train in X_train]
# print(distances)
# k=2
# k_indices = np.argsort(distances)[:k]   
# print(k_indices)
# lista = ["maca", "banana", "laranja", "maca", "maca"] 
# unique, counts = np.unique(lista, return_counts=True)
# print("Elementos unicos", unique)
# print("Contagem", counts)
# print(unique[np.argmax(counts)])"""