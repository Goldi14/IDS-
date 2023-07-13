import time
import numpy as np
from math import sqrt


from scipy.stats import pearsonr


import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from utils import save_generated_feats
from utils import load_generation, save_generation

from data import get_train_test_data
from utils import get_model
from utils import get_anova_scores

from model import train_models
import xgboost as xgb
from config import xgb_params 



X_train, Y_train, X_test, Y_test = get_train_test_data()




score = get_anova_scores(X_train,Y_train)



def countNoOfFeatures(chromosome):
    k = 0
    for gene in chromosome:
        k += gene
    return k


def computeMerit(chromosome):
    print("computing merit:")
    k = countNoOfFeatures(chromosome)
    x = X_train.iloc[:, chromosome].to_numpy()
    y = Y_train

    n = x.shape[1]
    Rcf = 0
    Rff = 0
    for i in range(n):
        for j in range(i+1, n):
            Rff += pearsonr(x[:, i], x[:, j])[0]

    Rff = Rff/((n*(n-1))/2)
    Rcf = 0.0
    for i in range(score.shape[0]):
        if(chromosome[i] == 1):
            Rcf += score[i][0]
    Rcf = Rcf/n

    print("Rcf: ", Rcf, "/", "Rff: ", Rff)

    merit = (k*Rcf)/sqrt(k+k*(k-1)*abs(Rff))
    return merit


def GMean(y_pred, y_true):
    TP, TN, FP, FN = 0, 0, 0, 0
    cnf_matrix = confusion_matrix(y_true, y_pred)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    sensitivity = TP/(TP+FN)  # true positive rate
    specificity = TN/(TN+FP)  # true negative rate

    g_mean = sqrt((sum(sensitivity)/len(sensitivity))
                  * (sum(specificity)/len(specificity)))
    return g_mean


def initilization_of_population(size, n_feat):
    population = []
    np.random.seed(42)                                  
    for i in range(size):
        chromosome = np.ones(n_feat, dtype=np.bool)
        chromosome[:np.int(0.3*n_feat)] = False         
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score(population, a):
    scores = []

    # varibale declare kiya
    merits = []
    g_means = []
    f_scores = []
    accs = []

    for chromosome in population:
        merit = computeMerit(chromosome)
    
        # model.fit(X_train.iloc[:, chromosome], Y_train)
        # predictions = model.predict(X_test.iloc[:, chromosome])

        #---------------- xgb model-------------------#
        dtrain = xgb.DMatrix(X_train.iloc[:, chromosome], label=Y_train)
        dvalid = xgb.DMatrix(X_test.iloc[:, chromosome], label=Y_test)  

        model = xgb.train(xgb_params, dtrain)
        preds = model.predict(dvalid)
        predictions = np.rint(preds)
        acc = sklearn.metrics.accuracy_score(Y_test, predictions)
        ####################################
        g_mean = GMean(predictions, Y_test)
        f_score = (a * merit) + ((1-a)*g_mean)

        #acc = accuracy_score(Y_test, predictions)

        scores.append(f_score)

        # variable me value append kari
        merits.append(merit)
        g_means.append(g_mean)
        f_scores.append(f_score)
        accs.append(acc)

    # variable se dictionry me convert kiya
    fitness_metrics = {
        'merit': merits,
        'g_mean': g_means,
        'f_score': f_scores,
        'acc': accs
    }

    scores, population = np.array(scores), np.array(population)
    inds = np.argsort(scores)  # sorting in ascending order

    # also returning fitness matrics
    return list(scores[inds][::-1]), list(population[inds, :][::-1]), fitness_metrics


def selection(pop_after_fit, selection_rate, pop_size):
    population_nextgen = []
    count = np.int(selection_rate*pop_size)                       
    for i in range(count):
        population_nextgen.append(pop_after_fit[i])
    return population_nextgen


def crossover(pop_after_sel, selection_rate, pop_size):
    print("----------------------Crossover operation------------------------")
    pop_nextgen = pop_after_sel
    count = (np.int((1-selection_rate)*pop_size))//2           
    for i in range(0, count, 2):
        new_par1 = []
        new_par2 = []
        child_1, child_2 = pop_nextgen[i], pop_nextgen[i+1]
        new_par1 = np.concatenate((child_1[:len(child_1)//2], child_2[len(child_1)//2:]))
        new_par2 = np.concatenate((child_1[len(child_1)//2:], child_2[:len(child_1)//2]))
        pop_nextgen.append(new_par1)
        pop_nextgen.append(new_par2)
    return pop_nextgen


def mutation(pop_after_cross, mutation_rate, n_feat):
    print("---------------------------Mutation operation-----------------------")
    mutation_range = np.int(mutation_rate*n_feat) 
    pop_next_gen = []
    np.random.seed(42)                          
    for n in range(0, len(pop_after_cross)):
        chromo = pop_after_cross[n]
        rand_posi = []
        for i in range(0, mutation_range):
            #pos = randint(0,n_feat-1)
            pos = np.random.randint(n_feat, size=1)[0]               

            rand_posi.append(pos)
        for j in rand_posi:
            chromo[j] = not chromo[j]
        pop_next_gen.append(chromo)
    return pop_next_gen


def generations(pop_size, n_feat, mutation_rate, selection_rate, a, n_gen, from_gen):     
    best_chromo = []
    best_score = []
    population_nextgen = initilization_of_population(pop_size, n_feat)

    #----------------------change------#

    scores = None
    pop_after_fit = None
    fitness_metrics = None
    pop_after_sel = None
    pop_after_cross = None

    if from_gen != 1:                                                              

        print(f"Need to load from prior run generation result : {from_gen-1} \n")     
        _generation_data = load_generation(gen_id=from_gen-1, constant=a)             

        scores = _generation_data["scores"]
        fitness_metrics = _generation_data["fitness_metrics"]
        population_nextgen = _generation_data["population_nextgen"]
        best_chromo = _generation_data["best_chromo"]
        best_score = _generation_data["best_score"]

                                                              
    #----------------------till here------#

    for i in range(from_gen, n_gen+1):                                               
        print(f"\n----------------Generation: {i} details---------------------\n")         
        t1 = time.time()

        scores, pop_after_fit, fitness_metrics = fitness_score(population_nextgen, a=a)
        print('Best score in generation ', i, ':', scores[:1])  # 2

        pop_after_sel = selection(pop_after_fit, selection_rate, pop_size)
        pop_after_cross = crossover(pop_after_sel, selection_rate, pop_size)
        population_nextgen = mutation(pop_after_cross, mutation_rate, n_feat)
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])

        print("\n==============================================================\n")
        print(f"saving generation : {i}")
        _generation_data = {
            "scores": scores,
            "fitness_metrics": fitness_metrics,
            "population_nextgen": population_nextgen,
            "best_chromo": best_chromo,                            #pop_after_fit[0],
            "best_score": best_score,                              #scores[0],
        }

        save_generation(gen_id=i, constant=a, data=_generation_data)
        print(f"generation : {i} : constant : {a} saved successfully!")

        if i in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:                   
            chromo_df_bc,score_bc = best_chromo, best_score
            _X_train_df, _Y_train_df, _X_test_df, _Y_test_df = save_generated_feats(i, score_bc, chromo_df_bc, 
                                                                                    X_train, Y_train, X_test, Y_test)

        print(f"Genearation taken : {time.time() - t1} seconds")
 
    return best_chromo, best_score
