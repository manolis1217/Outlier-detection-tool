#!/usr/bin/env python
# coding: utf-8

# In[1]:


import my_functions as my_func
import numpy as np  
import warnings
import time
import pandas as pd
import random


# In[2]:


# function set_params()
# chunk 1
class Detector():
    def __init__(self, method = 'loda'):
        self.method = method
        
    def get_data(self, path):
        """
        The imported csv/xlsx should have no missing values as well as no 
        empty rows or columns.
        """
        path = path.replace('\\','/') # reverse slash, as the backslash is not acceptable in python
        path = path.translate({ord("\""):None}) # removes "" from the beggining and the end of the path
        if path[-4:]=='xlsx':                                                   
            ds = pd.read_excel(path)
        elif path[-3:]=='csv':
            ds = pd.read_csv(path)
        else:
            raise Exception("Not acceptable type of file\n"
                          "Only .csv or .xlsx files can be inputted")
        return (ds)

    def detect(self, ds, train_split = 0.5, con = 0.1, n = 3, th = 0.3 ):
        method = self.method
        train_split = float(train_split)
        con = float(con)
        n = int(n)
        th = float(th)
        if method != 'ngram' and method != 'AE' and method != 'IF' and method != 'loda':
            raise Exception("The available methods are: 'loda', 'AE', 'IF', 'ngram'.\n"
                           "Please choose one of the available methods.")
        if method!='ngram':
            if train_split > 1 or train_split < 0:
                raise Exception("The training split parameters represents the percentage \n"
                               "of the dataset that is used for training. Therefore, allowed \n"
                               "values are between 0 and 1")

        if method == 'loda' or method == 'AE' or method == 'IF':
            # Drop any attriibute that is not numerical
            for col in ds.columns:
                if type(ds[col][1])!=np.int64 and type(ds[col][1])!=np.float64:
                    ds.drop(col,axis=1,inplace=True)
        else:
            # Drop any attribute that is not string
            for col in ds.columns:
                if type(ds[col][0]) != str:
                    ds.drop(col,axis=1,inplace=True)
        if method != 'ngram':
            sample_size = int(train_split*len(ds))
            indices = random.sample(range(0, len(ds)), sample_size)
            sample = ds.iloc[indices]
            
        size = "big" if len(ds) > 10000 else "small"
        config = my_func.read_config()
        epochs = int(config[size]['epochs'])
        activation_function = config[size]['activation_function']
        batch_size = int(config[size]['batch_size'])
        if config[size]['lay'] == 'long':
            layers=[64,32,16,1,16,32,64]
        if config[size]['lay'] == 'short':
            layers=[32,16,1,16,32]
        estimators=config[size]['estimators']
        
        if method == 'AE':
            warnings.filterwarnings("ignore")
            start=time.time()
            ds_pred=my_func.autoencoder_detection(ds ,sample, layers=layers,epo=epochs,
                                                  activation_function=activation_function,
                                                  batch=batch_size,cont=con)

            finish=time.time()
            print('It took ', finish-start,' seconds')
        elif method=='IF':
            start=time.time()
            ds_pred=my_func.IsolationForest_detection(ds, sample, estimators=estimators)
            finish=time.time()
            print('It took ', finish-start,' seconds')
        elif method=='loda':
            start=time.time()
            ds_pred=my_func.loda_detection(ds, sample)
            finish=time.time()
            print('It took ', finish-start,' seconds')
        elif method=='ngram' and att_ranking==0 :
            start=time.time()
            ds_pred=my_func.ngrams_detection(ds, n, th)
            finish=time.time()
            print('It took ', finish-start,' seconds')
        return (ds_pred)


class Ranker():
    def __init__(self,  method='loda', att_ranking = 0, ind = 0, th = 0.1):
        self.att_ranking = att_ranking
        self.ind = ind
        self.th = th
        self.method= method
    def rank(self, ds_pred, ds, func = 'mean'): 
        att_ranking = int(self.att_ranking)
        ind = int(self.ind)
        th = float(self.th)
        method = self.method
        if att_ranking == 0:                        # th is tuples in the top % interval score
            # ranking by tuples
            if method != 'ngram':
                if ind == 0:
                    final=my_func.count_anomalies(ds_pred,method)
                    range_of_val=max(final['Ranking_count'])-min(final['Ranking_count'])
                    limit=(1-th)*range_of_val+min(final['Ranking_count'])
                    return [final[final['Ranking_count']>limit].index, len(final[final['Ranking_count']>limit].index)/len(ds) ]
                elif ind==1:
                    edges=my_func.get_edges(ds_pred,method)
                    nodes=my_func.get_nodes(ds_pred,method)
                    G=my_func.build_network(nodes,edges)
                    final=my_func.pagerank_score(G,ds_pred)
                    range_of_val=max(final['PageRank score'])-min(final['PageRank score'])
                    limit=(1-th)*range_of_val+min(final['PageRank score'])
                    return [final[final['PageRank score']>limit].index, len(final[final['PageRank score']>limit].index)/len(ds)]
            else:
                return (my_func.ngrams_ranking(ds_pred, th))
        else:
            # ranking by attribute
            if method!='ngram':
                return (my_func.dataframe_score(ds_pred,ds,method,func, th))
            if method=='ngram':
                score=my_func.ngram_score_att(ds,n,th)
                norm=my_func.maximum_score_att(ds,n)
                df=pd.DataFrame({'normalized_score':score/norm})
                if func=='mean':
                    score = np.mean(score/norm)
                if func=='median':
                    score = np.median(score/norm)

                return [df[df.normalized_score>1-th].index, score]

