"""
Author: Saurabh Annadate

This script trains fasttext models to classify the text into categories
"""

import logging
import os
import pandas as pd
import yaml
import logging
import datetime
import pickle

from tqdm import tqdm

import fasttext

from Scripts.text_analytics_helpers import pre_process_data
from Scripts.helpers import load_data

from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
   
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

logger = logging.getLogger()

def get_cross_val_scores(X,y, lr, wordNgrams, dim, epoch=10):
    """
    Runs a 5-fold cross validation model fit and returns the results
        
    Args:
        X(Pandas Series): Text
        y(Pandas Series): Labels
    
    Returns:
        List of []
    
    """
    
    accuracy_list = []
    precision_list = []
    recall_list = []
    fscore_list = []
    
    labels = ["__label__" + str(ele) for ele in y]
    texts = [" " + ele + "\n" for ele in X]
    
    kfold = KFold(n_splits=5, random_state=12345)
    for train,test in kfold.split(X, y):
        TrainCorpus = [i + j for i, j in zip([labels[i] for i in train], [texts[i] for i in train])]
        TestCorpus = [i + j for i, j in zip([labels[i] for i in test], [texts[i] for i in test])]
        
        TrainCorpus = ''.join(TrainCorpus)
        TestCorpus = ''.join(TestCorpus)
        
        with open('Outputs/fast.train', 'w') as f:
            f.write(TrainCorpus)
            f.close()

        with open('Outputs/fast.test', 'w') as f:
            f.write(TestCorpus)
            f.close()
        
        base = "~/fastText-0.9.1/fasttext predict "

        model = fasttext.train_supervised(input="Outputs/fast.train", wordNgrams=wordNgrams, lr=lr, loss='ns', dim = dim, epoch = epoch, verbose = 0, neg = 25)
        model.save_model("Outputs/fastText_CV_model.bin")
        
        commandtext = base + "Outputs/fastText_CV_model.bin" + " Outputs/fast.test 1 > Outputs/fastTextPreds.txt"
        os.system(commandtext)
        
        f = open("Outputs/fastTextPreds.txt", 'r')
        y_pred = f.read().splitlines()
        f.close()
        
        
        rest_sc = precision_recall_fscore_support([labels[i] for i in test],y_pred)
        
        accuracy_list.append(accuracy_score([labels[i] for i in test],y_pred))
        precision_list.append(rest_sc[0][1])
        recall_list.append(rest_sc[1][1])
        fscore_list.append(rest_sc[2][1])
 
    return([sum(accuracy_list)/5, sum(precision_list)/5, sum(recall_list)/5, sum(fscore_list)/5])
        

def fit_fasttext():
    """
    Fits a fasttext classification model on the data to model the review

    Args:
        None
    
    Returns:
        None
    """

    logger.debug("Running the fit_fasttext function now")

    #Loading the configuration
    with open(os.path.join("config","config.yml"), "r") as f:
        config = yaml.safe_load(f)

    #Loading and pre processing the data
    logger.debug("Loading and pre processing the data")
    train_df = load_data(config["load_data"]["train_file"])
    train_df = pre_process_data(train_df, resample = True, resample_count = 500000)

    #Defining Parameters
    lr = [0.01, 0.1, 0.5, 1] 
    wordNgrams = [1,2]
    dim = [100, 500, 1000, 5000]
    
    lr_list = []
    wordNgrams_list = []
    dim_list = []
    
    accuracy_list = []
    prec_list = []
    recall_list = []
    fscore_list = []
    
    t0 = datetime.datetime.now()
    
    #Running fasttext models:
    for i in tqdm(lr, desc = "Learning Rate"):
        for j in tqdm(wordNgrams, desc = "Word Grams"):
            for k in tqdm(dim, desc = "Dimensions"):

                res = get_cross_val_scores(train_df['Review'],train_df['Ratings'], lr=i, wordNgrams = j, dim = k)
                
                lr_list.append(i)
                wordNgrams_list.append(j)
                dim_list.append(k)
                
                accuracy_list.append(res[0])
                prec_list.append(res[1])
                recall_list.append(res[2])
                fscore_list.append(res[3])

    logger.info("Grid Search performed in {}".format(str(datetime.datetime.now()-t0)))
    
    #Writing results
    res_df = pd.DataFrame(list(zip(lr_list, wordNgrams_list, dim_list, accuracy_list, prec_list, recall_list, fscore_list)), columns = ["lr_list", "wordNgrams_list", "dim_list", "accuracy_list", "prec_list", "recall_list", "fscore_list"])
    res_df.to_csv("Outputs/fasttext_results.csv")
    
    return