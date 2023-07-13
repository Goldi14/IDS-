import numpy as np

from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

import seaborn as sns
import matplotlib.pyplot as plt



def DirectMetrics(actual, predicted):
    
    cr = classification_report(actual, predicted)           
    print("Classification_Report : ")
    print(cr)
  
    Accuracy = accuracy_score(actual, predicted)
    print("Accuracy: %.2f%%" % (Accuracy * 100.0))
    
    Precision = precision_score(actual, predicted, average = 'weighted')
    Recall = recall_score(actual, predicted, average = 'weighted')
    F1_Score = f1_score(actual, predicted, average = 'weighted')
    
    return {"Direct_Precision" : Precision,
           "Direct_Recall" : Recall,
           "Direct_F1-Score" : F1_Score}


def ComputeMetrics(actual , predicted):
    TP, TN, FP, FN = 0, 0, 0, 0
    
    cm = confusion_matrix(actual, predicted)

    
    FP = cm.sum(axis = 0) - np.diag(cm)
    FN = cm.sum(axis = 1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    sensitivity = TP/(TP+FN)
    avg_sensitivity = sum(sensitivity)/len(sensitivity)

    specificity = TN/(TN+FP)
    avg_specificity = sum(specificity)/len(specificity)

    precision = TP/(TP+FP)
    avg_precision = sum(precision)/len(precision)
    
    recall = TP/(TP+FN)
    avg_recall = sum(recall)/len(recall)
    
    f1_score = (2*recall*precision)/(recall + precision)
    avg_f1_score = sum(f1_score)/len(f1_score)
    
    FAR = FP/(FP+TN)
    avg_FAR = sum(FAR)/len(FAR)
    
    return {"Sensitivity " : avg_sensitivity,
           "Specificity " : avg_specificity,
           "Precision " : avg_precision,
           "Recall " : avg_recall,
           "F1_Score " : avg_f1_score,
           "FAR " : avg_FAR}          

