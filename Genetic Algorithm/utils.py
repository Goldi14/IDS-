import pickle
import numpy as np
import modin.pandas as pd
from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

def get_anova_scores(X_train,Y_train):
    # configure to select all features
    fs = SelectKBest(score_func=f_classif, k='all')
    # learn relationship from training data
    fs.fit(X_train, Y_train)
    # transform train input data
    X_train_fs = fs.transform(X_train)

    #calc Anova scores
    score = fs.scores_
    score = np.asarray(score).reshape(len(score),-1)

    #Scale AnovaScore
    mimxScale = MinMaxScaler()
    score = mimxScale.fit_transform(score)
    
    return score


def get_model():                                           
    return XGBClassifier(learning_rate = 0.1,
          n_estimators=4, 
          max_depth=3,
          nthread=4,
          seed = 1)

def save_generation(gen_id, constant, data):
    with open(str(gen_id) + "_" + str(constant) + "_.pickle", "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

def load_generation(gen_id, constant):
    with open(str(gen_id) + "_" + str(constant) + "_.pickle", 'rb') as handle:
        data = pickle.load(handle)

    return data

def save_generated_feats(gen_id, score_bc, chromo_df_bc, X_train, Y_train, X_test, Y_test):
    ind = 0
    l = len(score_bc)
    _max = 0

    for i in range(l):
      if(_max < score_bc[i]):
        _max = score_bc[i]
        ind = i

    chromosome = chromo_df_bc[ind]

    print(chromosome)

    X_train_df = X_train.iloc[:, chromosome]
    X_test_df = X_test.iloc[:, chromosome]

    Y_train_df = pd.DataFrame(Y_train, columns =['attack'])
    Y_test_df = pd.DataFrame(Y_test, columns = ['attack'])

    train_df = pd.concat([X_train_df, Y_train_df] , axis = 1)
    test_df = pd.concat([X_test_df, Y_test_df] , axis = 1)

    train_df.to_csv(f"train_df_after_fs_{gen_id}.csv",index=False)
    test_df.to_csv(f"test_df_after_fs_{gen_id}.csv",index = False)
    
    return X_train_df, Y_train_df, X_test_df, Y_test_df

def save_gen_model_res(gen_id, data):
    with open(f'gen_model_res_{str(gen_id)}_.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_gen_model_res(gen_id) :    
    with open(f'gen_model_res_{str(gen_id)}_.pickle', 'rb') as handle:
        return pickle.load(handle)