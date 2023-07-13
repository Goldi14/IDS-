import numpy as np
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC                             
from sklearn.neighbors import KNeighborsClassifier      
from sklearn.ensemble import RandomForestClassifier     
from xgboost import XGBClassifier

import xgboost as xgb
from config import xgb_params 

    
from metrics_helper import DirectMetrics
from metrics_helper import ComputeMetrics

from utils import save_gen_model_res

    
def train_models(gen_id, X_train_df, Y_train_df, X_test_df, Y_test_df):

    _dict = { model_type : {"direct_metrics" : None, "metrics" : None} for model_type in ["knn", "rf", "dt", "xgb", "svm"]}  

    
    
    #KNN                                                                  
    model_knn = KNeighborsClassifier()
    model_knn.fit(X_train_df, Y_train_df)

    Y_pred_knn = model_knn.predict(X_test_df)
    predictions_knn = [round(value) for value in Y_pred_knn]

    direct_metrics_knn = DirectMetrics(Y_test_df, predictions_knn)
    metrics_knn = ComputeMetrics(Y_test_df, predictions_knn)
    
    
    #Random Forest                                                  
    model_rf = RandomForestClassifier(n_estimators = 100)            #change no of estimators 
    model_rf.fit(X_train_df, Y_train_df)

    Y_pred_rf = model_rf.predict(X_test_df)
    predictions_rf = [round(value) for value in Y_pred_rf]

    direct_metrics_rf = DirectMetrics(Y_test_df, predictions_rf)
    metrics_rf = ComputeMetrics(Y_test_df, predictions_rf)
    
    
    #Decision Tree
    model_dt = DecisionTreeClassifier()
    model_dt.fit(X_train_df, Y_train_df)

    Y_pred_dt = model_dt.predict(X_test_df)
    predictions_dt = [round(value) for value in Y_pred_dt]

    direct_metrics_dt = DirectMetrics(Y_test_df, predictions_dt)
    metrics_dt = ComputeMetrics(Y_test_df, predictions_dt)


    #----------------xgb model-------------------#
    dtrain = xgb.DMatrix(X_train_df, label=Y_train_df)
    dvalid = xgb.DMatrix(X_test_df, label=Y_test_df)  

    model = xgb.train(xgb_params, dtrain)
    preds = model.predict(dvalid)
    predictions_xgb = np.rint(preds)
    acc = sklearn.metrics.accuracy_score(Y_test_df, predictions_xgb)

    direct_metrics_xgb = DirectMetrics(Y_test_df, predictions_xgb)
    metrics_xgb = ComputeMetrics(Y_test_df, predictions_xgb)
       
        
    #SVM                                                                
    model_svm = SVC()
    model_svm.fit(X_train_df, Y_train_df)

    Y_pred_svm = model_svm.predict(X_test_df)
    predictions_svm = [round(value) for value in Y_pred_svm]

    direct_metrics_svm = DirectMetrics(Y_test_df, predictions_svm)
    metrics_svm = ComputeMetrics(Y_test_df, predictions_svm)


    ################-----------------------------------------------------####################

    
    _dict["knn"]["direct_metrics"] = direct_metrics_knn                    
    _dict["knn"]["metrics"] = metrics_knn

    _dict["rf"]["direct_metrics"] = direct_metrics_rf                     
    _dict["rf"]["metrics"] = metrics_rf
    
    _dict["dt"]["direct_metrics"] = direct_metrics_dt
    _dict["dt"]["metrics"] = metrics_dt

    _dict["xgb"]["direct_metrics"] = direct_metrics_xgb
    _dict["xgb"]["metrics"] = metrics_xgb
       
    _dict["svm"]["direct_metrics"] = direct_metrics_svm                  
    _dict["svm"]["metrics"] = metrics_svm
    
    #save the values
    save_gen_model_res(gen_id, _dict)
    print("*" * 50)
    print("*" * 50)
    print(_dict)
    print("*" * 50)
    
