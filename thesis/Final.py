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


def main():
    # chunk 1
    """
    The imported csv/xlsx should have no missing values as well as no 
    empty rows or columns.
    """
    in_path= input('Import the path of the file to be tested on anomalies\n')
    path=in_path.replace('\\','/') # reverse slash, as the backslash is not acceptable in python
    path=path.translate({ord("\""):None}) # removes "" from the beggining and the end of the path
    if path[-4:]=='xlsx':                                                   
        ds=pd.read_excel(path)
    elif path[-3:]=='csv':
        ds=pd.read_csv(path)
    else:
        raise Exception("Not acceptable type of file\n"
                      "Only .csv or .xlsx files can be inputted")
    
    # chunk 1-2
    
    att_ranking=input('Do you want to rank the dataset by rows or by attribute?\n'
                     '0: Ranking by rows\n'
                     '1: Ranking by attributes\n'
                     'Enter the variable:\n\n')
    att_ranking=int(att_ranking)
    if att_ranking!=0 and att_ranking!=1:
        raise Exception('Invalid input')
    # chunk 2
    """
    Import parameters for the program such as: 
    - method to be used for outlier detection
    - percentage of dataset to be used as training
    """
    method=input('Enter the desired method. \n'
                         "available methods are: 'ngram', 'AE', 'IF', 'loda'\n"
                         'Enter the parameter:')
    if method!='ngram' and method!='AE' and method!='IF' and method!='loda':
        raise Exception("The available methods are: 'loda', 'AE', 'IF', 'ngram'.\n"
                       "Please choose one of the available methods.")
    if method!='ngram':
        train_split=float(input('Enter the training split of the dataset:'))

    if method!='ngram':
        if train_split>1 or train_split<0:
            raise Exception("The training split parameters represents the percentage \n"
                           "of the dataset that is used for training. Therefore, allowed \n"
                           "values are between 0 and 1")
    # chunk 3       
    if method=='loda' or method=='AE' or method=='IF':
        # Drop any attriibute that is not numerical
        for col in ds.columns:
            if type(ds[col][1])!=np.int64 and type(ds[col][1])!=np.float64:
                ds.drop(col,axis=1,inplace=True)

    else:
        # Drop any attribute that is not string
        for col in ds.columns:
            if type(ds[col][0])!=str:
                ds.drop(col,axis=1,inplace=True)

    # chunk 4
    if method!='ngram':
        sample_size=int(train_split*len(ds))
        indices=random.sample(range(0, len(ds)), sample_size)
        sample=ds.iloc[indices]

    # chunk 5
    if method=='AE':
        print('Autoencoder method initiated for outlier detection')
        new_param=list(input('Further parameters are needed to be established for the Autoencoder\n'
                             'Please enter them in the following format: (epochs, activation function, batch size, contamination, fast)\n'
                             'Enter the parameters:').split(','))
        epochs, act_function, bat, con,f=int(new_param[0]), new_param[1], int(new_param[2]), float(new_param[3]), int(new_param[4])
        warnings.filterwarnings("ignore")

        start=time.time()
        ds_pred=my_func.autoencoder_detection(ds ,sample,epo=epochs,
                                              activation_function=act_function, batch=bat,cont=con,fast=f)

        finish=time.time()
        print('The outlier detection with '+ method+' is completed.\n')
        print('It took ', finish-start,' seconds')
    elif method=='IF':
        print('Isolation Forest method initiated for outlier detection')
        n_estim=int(input('Number of estimators needs to be established for the Isolation Forest\n'
                             'Enter the parameter:'))

        start=time.time()
        ds_pred=my_func.IsolationForest_detection(ds, sample, estimators=n_estim)
        finish=time.time()
        print('The outlier detection with '+ method+' is completed.\n')
        print('It took ', finish-start,' seconds')
    elif method=='loda':
        print('Lightweight online detection of anomalies initiated for outlier detection')

        start=time.time()
        ds_pred=my_func.loda_detection(ds, sample)
        finish=time.time()
        print('The outlier detection with '+ method+' is completed.\n')
        print('It took ', finish-start,' seconds')
    elif method=='ngram' and att_ranking==0 :
        n, th=input('The number of grams needs to be given as well as the threshold for outlying values\n'
                   'Enter the parameters in the following format (n, th)\n'
                   'e.g. n=2, bigrams will be collected from the values\n'
                   'The threshold determines the acceptable deviance from the average of the frequencies of ngrams\n'
                   'Enter the parameters:').split(',')
        n=int(n)
        th=float(th)
        print(n,'-grams method is initiated for outlier detection')
        start=time.time()
        ds_pred=my_func.ngrams_detection(ds, n, th)
        finish=time.time()
        print('The outlier detection with '+ method+' is completed.\n')
        print('It took ', finish-start,' seconds')

    

    # chunk 6
    if att_ranking==0:
        # ranking the tuples
        if method!='ngram':
            ind,th=input('Enter the indicator of the ranking method and the threshold of the most anomalous tuples \n'
                    'in the following format: (ind,th).\n'
                    'Indicator can be 0 or 1,\n 0: Count of anmolies \n 1: PageRank Algorithm\n'
                    'The threshold variable determines which tuples index will be returned.\n'
                    'e.g. threshold=0.1, returns the tuples with the top 10% score \n'
                    'Enter the variables:').split(',')
            ind=int(ind)
            th=float(th)
            if ind==0:
                final=my_func.count_anomalies(ds_pred,method)
                range_of_val=max(final['Ranking_count'])-min(final['Ranking_count'])
                limit=(1-th)*range_of_val+min(final['Ranking_count'])
                print('The anomalous tuples that exceed the threshold of',1-th,'are:')
                print(final[final['Ranking_count']>limit].index)
            elif ind==1:
                edges=my_func.get_edges(ds_pred,method)
                nodes=my_func.get_nodes(ds_pred,method)
                G=my_func.build_network(nodes,edges)
                final=my_func.pagerank_score(G,ds_pred)
                range_of_val=max(final['PageRank score'])-min(final['PageRank score'])
                limit=(1-th)*range_of_val+min(final['PageRank score'])
                print('The ',th*100,'% most anomalous tuples are:')
                print(final[final['PageRank score']>limit].index)
        else:
            threshold=float(input('The threshold of most anomlaous value must be given\n'
                           'The threshold must belong in [0,1]\n'
                            'e.g. threshold=1 -> returns the whole dataset\n'
                           'Enter the threshold:'))

            print(my_func.ngrams_ranking(ds_pred, threshold))
    else:
        func=input("Select a function to calculate the score of the dataframe\n"
                      'Available functions are: mean, median\n'
                      'Enter the function:').lower()
        if method!='ngram':
            print(my_func.dataframe_score(ds_pred,ds,method,func))
        if method=='ngram':
            n, th=input('The number of grams needs to be given as well as the threshold for outlying values\n'
                   'Enter the parameters in the following format (n, th)\n'
                   'e.g. n=2, bigrams will be collected from the values\n'
                   'The threshold determines the acceptable deviance from the average of the frequencies of ngrams\n'
                   'Enter the parameters:').split(',')
            n=int(n)
            th=float(th)
            score=my_func.ngram_score_att(ds,n,th)
            norm=my_func.maximum_score_att(ds,n)
            if func=='mean':
                print (np.mean(score/norm))
            if func=='median':
                print (np.median(score/norm))
            
    restart=input('Do you want to restart? (y/n)').lower()
    if restart=='y':
        main()
    else:
        exit()
    
# In[3]:
main()


