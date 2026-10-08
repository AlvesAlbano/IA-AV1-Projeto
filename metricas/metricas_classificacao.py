import numpy as np

def matriz_confusao(y_pred,y_test):
    classes = np.unique(
        np.concatenate((y_pred,y_test))
    )

    matriz_confu = np.zeros(
        (len(classes),len(classes)),dtype=int
    )

    for real, previsto in zip(y_test,y_pred):
        i = np.where(classes == real)[0][0]
        j = np.where(classes == previsto)[0][0]

        matriz_confu[i,j] += 1
        
    return classes, matriz_confu

def accuracy(VP,FN,FP,VN):
    return (VN + VP) / (VP + FN + FP + VN)

def f1_score(precision,recall):
    return 2 * ((precision * recall)/(precision + recall)) 

def precision(VP,FP):
    return VP / (FP + VP)

def recall(VP,FN):
    return VP / (FN + VP)