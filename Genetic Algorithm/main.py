import warnings

import numpy as np

#custom imports
from ga_helper import generations

#np.random.seed(0)                              #commenting
warnings.filterwarnings("ignore")


from data import get_train_test_data

#load the data
X_train, Y_train, X_test, Y_test = get_train_test_data()


chromo_df_bc,score_bc = generations(pop_size=50,
                        n_feat=X_train.shape[1],
                        mutation_rate=0.5,
                        selection_rate=0.8,
                        a = 0.4, 
                        n_gen=100,
                        from_gen=1)                                 

