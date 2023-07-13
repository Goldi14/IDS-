import numpy as np
import modin.pandas as pd


def LabelAttack(Y):
  attack = []
  for i in range(Y.shape[0]):
    k = 0
    for j in range(Y.shape[1]):
      if(Y[i][j] == 1):
        k = j
        break
    attack.append(k)
  return attack


def get_train_test_data():
    test_df = pd.read_csv(r'test_dataset.csv')             #.sample(1024)
    train_df = pd.read_csv(r'balanced_train_dataset.csv')  #.sample(1024)

    label = ['dos', 'normal', 'probe', 'r2l', 'u2r']
    #label = ['Analysis','Backdoor','DoS','Exploits','Fuzzers','Generic','Normal','Reconnaissance','Shellcode','Worms']

    X_train = train_df.iloc[:,:train_df.shape[1]-5]
    Y_train = train_df.iloc[:,-5:].to_numpy()

    X_test = test_df.iloc[:,:test_df.shape[1]-5]
    Y_test = test_df.iloc[:,-5:].to_numpy()
   

    Y_train = np.asarray(LabelAttack(Y_train)).reshape(-1,1)
    Y_test = np.asarray(LabelAttack(Y_test)).reshape(-1,1)


    return X_train, Y_train, X_test, Y_test

