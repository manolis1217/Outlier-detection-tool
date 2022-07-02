import sys
import subprocess
# ensure the needed libraries are installed in the system
subprocess.check_call([sys.executable, '-m', 'pip', 'install','pandas'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','numpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','networkx'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','pyod'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','scipy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','sklearn'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','tensorflow'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install','openpyxl'])


import pandas as pd
import numpy as np
import networkx as nx
import random
import pyod
import configparser
import scipy.io
import time
from collections import Counter
from sklearn.ensemble import IsolationForest
from pyod.models import loda
from pyod.models import auto_encoder

def ngrams(dataframe, n_grams=3, n_att=1):
    """
    Description:
    -------------
        Has an input of a dataframe and returns n-grams of the inputted text
    
    Parameters:
    -----------
        dataframe: input dataframe, each attribute of it will be tokenized
        n_grams: define the number of characters in each token
        n_att: define the number of attributes that will be examined at once
    
    Notes:
    ------
        n_att can have values in [1,3], otherwise it will return an error
        The result of the function depends the n_att variable.
        The result is a list of lists and can be interpreted as:
        Given that n_att=3, grams[n][m][l] is the list of ngrams of n,m,l 
        attributes concatanated together. arrange n,m,l from lower to higher
    """
    if (n_att== 0) or (n_att> 3):
        raise Exception("Not acceptable value")
    if n_att == 1:
        grams = []
        for att in dataframe.columns:
            grams_of_att = []
            for text in dataframe[att]:
                text = str(text)
                length = len(text)
                first_char = 0
                while first_char <= (length-n_grams):
                    grams_of_att.append(text[first_char:first_char+n_grams])
                    first_char += 1
            grams.append(grams_of_att)
    elif n_att == 2:
        grams = []
        for att_i in range(len(dataframe.columns)):
            grams_of_i2 = []
            for att2_i in range(att_i+1,len(dataframe.columns)):
                grams_of_text = []
                for text1,text2 in zip(dataframe.iloc[:,att_i],dataframe.iloc[:,att2_i]):
                    text1,text2 = str(text1),str(text2) # cast into string to concatanate
                    text = text1 + text2 # concatanation
                    length=len(text)
                    first_char = 0
                    while first_char <= (length-n_grams):
                        grams_of_text.append(text[first_char:first_char+n_grams])
                        first_char += 1
                grams_of_i2.append(grams_of_text)
            grams.append(grams_of_i2)
    elif n_att == 3:
        grams = []
        for att_i in range(len(dataframe.columns)):
            grams_of_i2 = []
            for att2_i in range(att_i+1,len(dataframe.columns)):
                grams_of_i3 = []
                for att3_i in range(att2_i+1,len(dataframe.columns)):
                    grams_of_text = []
                    for text1,text2,text3 in zip(dataframe.iloc[:,att_i],dataframe.iloc[:,att2_i],dataframe.iloc[:,att3_i]):
                        text1,text2,text3 = str(text1),str(text2),str(text3) 
                        text = text1 + text2 + text3 
                        length = len(text)
                        first_char = 0
                        while first_char <= (length-n_grams):
                            grams_of_text.append(text[first_char:first_char+n_grams])
                            first_char += 1
                    grams_of_i3.append(grams_of_text)
                grams_of_i2.append(grams_of_i3)
            grams.append(grams_of_i2)
    return (grams)
        

    
def predictions_in_df(predictions,n,trigger=-1):
    """
    Description:
    ------------
    Returns a dataframe with the anomalous tuple and the attributes that the anomaly occurs in
    
    Parameters:
    -----------
    predictions: list/array with the predictions for each data point
    n: number of attributes that are being examined for anomaly
    trigger: value that represents the outlier in the list/array
    
    Returns:
    -------
    A dataframe with one column for each of the following:
        attribute that takes part in the anomaly
        index of the anomalous tuple
    """
    if n == 2:
        _tuple,_att1,_att2 = [],[],[]
        for i,arr in enumerate(predictions):
            for j,arr1 in enumerate(arr):
                for l,element in enumerate(arr1):
                    if element == trigger:
                        _tuple.append(l)
                        _att1.append(i)
                        _att2.append(j)
        anomalies = pd.DataFrame()
        anomalies['1st_Attribute'] = _att1
        anomalies['2st_Attribute'] = _att2
        anomalies['tuple'] = _tuple
    if n == 3:
        _tuple,_att1,_att2,_att3 = [],[],[],[]
        for i,arr in enumerate(predictions):
            for j,arr1 in enumerate(arr):
                for k,arr2 in enumerate(arr1):
                    for l,element in enumerate(arr2):
                        if element == trigger:
                            _tuple.append(l)
                            _att1.append(i)
                            _att2.append(j)
                            _att3.append(k)
        anomalies = pd.DataFrame()
        anomalies['1st_Attribute'] = _att1
        anomalies['2st_Attribute'] = _att2
        anomalies['3st_Attribute'] = _att3
        anomalies['tuple'] = _tuple
    return(anomalies)

def single_outliers(preds,dataframe,method):
    if method != 'loda' and method != 'IF' and method != 'AE':
        raise Exception("Unknown method, "
                       "'single_outliers' works only with 'AE','IF' and 'loda'")
    met = str(method)
    single = []
    for i in range(len(dataframe)):
        _tuple = []
        for j in range(len(preds)):
            if preds[j][i] == 1:
                _tuple.append(j)
        single.append(_tuple)
    dataframe['single_'+met] = single
    dataframe = dataframe.rename(columns={'single': 'single_'+met})

    return (dataframe)
    
def outliers_pairs(preds,dataframe,method):
    if method!='loda' and method!='IF' and method!='AE':
        raise Exception("Unknown method, "
                       "'outliers_pairs' works only with 'AE','IF' and 'loda'")
    if method == 'AE' or method == 'loda':
        trig = 1
    elif method == 'IF':
        trig =- 1
    met = str(method)
    met = str(method)
    df = predictions_in_df(preds,2,trig)
    pairs = []
    _tup = []
    for i in df.tuple.unique():
        string=''
        for j in range(len(df[df.tuple == i].iloc[:,0])):
            string += str(df[df.tuple == i].iloc[j,0])
            string += str(df[df.tuple == i].iloc[j,0] + df[df.tuple == i].iloc[j,1]+1)+','
        pairs.append(string)
        _tup.append(i)
    for i in range(len(pairs)):
        temp = pairs[i].split(',')
        temp.pop()
        pairs[i] = temp
    pair_df = pd.DataFrame()
    pair_df['pairs'] = pairs
    pair_df['tuple'] = _tup
    dataframe['index'] = dataframe.index
    dataframe = dataframe.merge(pair_df,left_on = 'index',right_on = 'tuple',how = 'left')
    dataframe.drop(['tuple','index'],axis=1,inplace=True)
    dataframe['pairs'] = dataframe['pairs'].replace(np.nan, '')
    dataframe=dataframe.rename(columns = {'pairs': 'pairs_'+met})
    return (dataframe)



def outliers_triplets(preds,dataframe,method):
    if method!='loda' and method!='IF' and method!='AE':
        raise Exception("Unknown method, "
                       "'outliers_triplets' works only with 'AE','IF' and 'loda'")
    met = str(method)
    met = str(method)
    if method == 'AE' or method == 'loda':
        trig = 1
    elif method == 'IF':
        trig =- 1
    df = predictions_in_df(preds,3,trig)
    triplets = []
    _tup = []
    for i in df.tuple.unique():
        string=''
        for j in range(len(df[df.tuple==i].iloc[:,0])):
            string += str(df[df.tuple == i].iloc[j,0])
            string += str(df[df.tuple == i].iloc[j,0] + df[df.tuple == i].iloc[j,1]+1)
            temp = df[df.tuple == i].iloc[j,0] + df[df.tuple == i].iloc[j,1]+1
            string += str(temp+1+df[df.tuple==i].iloc[j,2])+','
        triplets.append(string)
        _tup.append(i)
    for i in range(len(triplets)):
        temp = triplets[i].split(',')
        temp.pop()
        triplets[i] = temp
    triplets_df = pd.DataFrame()
    triplets_df['triplets'] = triplets
    triplets_df['tuple'] = _tup
    dataframe['index'] = dataframe.index
    dataframe = dataframe.merge(triplets_df,left_on='index',right_on='tuple',how='left')
    dataframe.drop(['tuple','index'],axis=1,inplace=True)
    dataframe['triplets'] = dataframe['triplets'].replace(np.nan, '')
    dataframe = dataframe.rename(columns={'triplets': 'triplets_'+met})

    return (dataframe)

def Isolationforest_pairs(dataframe,train_split=0.2,estimators=100,contamination='auto',rnd_state=42):
    """
    Description:
    ------------
        Returns a prediction about each data point being an outlier basesd on the Isolation
    forest approach.
    
    Parameters
    ----------
        dataframe: The input to data to make predictions on
        train_split: percentage of the dataframe that will be used for training
        estimators: The number of trees in the ensemble
        contamination: the percentage of outliers in the dataset
        rnd_state: integer, to achieve reproducibility
    """
    clf = IsolationForest(n_estimators=estimators,contamination=contamination,random_state=rnd_state)
    sample_size = int(len(dataframe)*train_split)
    indices = random.sample(range(0, len(dataframe)), sample_size)
    sample = dataframe.iloc[indices]
    predictions = []
    for att1_i in range(len(dataframe.columns)):
        predections2 = []
        for att2_i in range(att1_i+1,len(dataframe.columns)):
            train=sample[[sample.columns[att1_i],sample.columns[att2_i]]]
            clf.fit(train)
            test=dataframe[[dataframe.columns[att1_i],dataframe.columns[att2_i]]]
            y_pred_test = clf.predict(test)
            predections2.append(y_pred_test)
        predictions.append(predections2)
    return(predictions)



def evaluation(True_label,Predictions):
        TP,TN,FP,FN = 0,0,0,0
        for i in range(len(thyroid)):
            if True_label[i] == Predictions[i]:
                if True_label[i] == 1:
                    TP += 1
                else:
                    TN += 1
            else:
                if True_label[i]==1:
                    FN += 1
                else:
                    FP += 1
        acc = (TP+TN)/(TP+TN+FP+FN)
        pre = TP/(TP+FP) 
        rec = TP/(TP+FN)  
        return (acc,pre,rec)


def pagerank(G, alpha=0.85, personalization=None,
			max_iter=100, tol=1.0e-6, nstart=None, weight='weight',
			dangling=None):
	"""Return the PageRank of the nodes in the graph.

	PageRank computes a ranking of the nodes in the graph G based on
	the structure of the incoming links. It was originally designed as
	an algorithm to rank web pages.

	Parameters
	----------
	G : graph
	A NetworkX graph. Undirected graphs will be converted to a directed
	graph with two directed edges for each undirected edge.

	alpha : float, optional
	Damping parameter for PageRank, default=0.85.

	personalization: dict, optional
	The "personalization vector" consisting of a dictionary with a
	key for every graph node and nonzero personalization value for each node.
	By default, a uniform distribution is used.

	max_iter : integer, optional
	Maximum number of iterations in power method eigenvalue solver.

	tol : float, optional
	Error tolerance used to check convergence in power method solver.

	nstart : dictionary, optional
	Starting value of PageRank iteration for each node.

	weight : key, optional
	Edge data key to use as weight. If None weights are set to 1.

	dangling: dict, optional
	The outedges to be assigned to any "dangling" nodes, i.e., nodes without
	any outedges. The dict key is the node the outedge points to and the dict
	value is the weight of that outedge. By default, dangling nodes are given
	outedges according to the personalization vector (uniform if not
	specified). This must be selected to result in an irreducible transition
	matrix (see notes under google_matrix). It may be common to have the
	dangling dict to be the same as the personalization dict.

	Returns
	-------
	pagerank : dictionary
	Dictionary of nodes with PageRank as value

	Notes
	-----
	The eigenvector calculation is done by the power iteration method
	and has no guarantee of convergence. The iteration will stop
	after max_iter iterations or an error tolerance of
	number_of_nodes(G)*tol has been reached.

	The PageRank algorithm was designed for directed graphs but this
	algorithm does not check if the input graph is directed and will
	execute on undirected graphs by converting each edge in the
	directed graph to two edges.

	
	"""
	if len(G) == 0:
		return {}

	if not G.is_directed():
		D = G.to_directed()
	else:
		D = G

	# Create a copy in (right) stochastic form
	W = nx.stochastic_graph(D, weight=weight)
	N = W.number_of_nodes()

	# Choose fixed starting vector if not given
	if nstart is None:
		x = dict.fromkeys(W, 1.0 / N)
	else:
		# Normalized nstart vector
		s = float(sum(nstart.values()))
		x = dict((k, v / s) for k, v in nstart.items())

	if personalization is None:

		# Assign uniform personalization vector if not given
		p = dict.fromkeys(W, 1.0 / N)
	else:
		missing = set(G) - set(personalization)
		if missing:
			raise NetworkXError('Personalization dictionary '
								'must have a value for every node. '
								'Missing nodes %s' % missing)
		s = float(sum(personalization.values()))
		p = dict((k, v / s) for k, v in personalization.items())

	if dangling is None:

		# Use personalization vector if dangling vector not specified
		dangling_weights = p
	else:
		missing = set(G) - set(dangling)
		if missing:
			raise NetworkXError('Dangling node dictionary '
								'must have a value for every node. '
								'Missing nodes %s' % missing)
		s = float(sum(dangling.values()))
		dangling_weights = dict((k, v/s) for k, v in dangling.items())
	dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]

	# power iteration: make up to max_iter iterations
	for _ in range(max_iter):
		xlast = x
		x = dict.fromkeys(xlast.keys(), 0)
		danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
		for n in x:

			# this matrix multiply looks odd because it is
			# doing a left multiply x^T=xlast^T*W
			for nbr in W[n]:
				x[nbr] += alpha * xlast[n] * W[n][nbr][weight]
			x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n]

		# check convergence, l1 norm
		err = sum([abs(x[n] - xlast[n]) for n in x])
		if err < N*tol:
			return x
	raise NetworkXError('pagerank: power iteration failed to converge '
						'in %d iterations.' % max_iter)
    
    

def get_edges(dataframe,method):
    links = []
    att = 'single_' + method
    att_pair = 'pairs_' + method
    att_triplets = 'triplets_' + method
    for i in range(len(dataframe)):
        for j in range(len(dataframe[att][i])):
            value_of_interest = dataframe.iloc[i,int(dataframe[att][i][j])]
            for l in range(len(dataframe[att][i])):
                if int(dataframe[att][i][j]) != int(dataframe[att][i][l]):
                    links.append([value_of_interest,dataframe.iloc[i,int(dataframe[att][i][l])]])
            for l in range(len(dataframe[att_pair][i])):
                for m in range(len(dataframe[att_pair][i][l])):
                    if int(dataframe[att_pair][i][l][m]) != int(dataframe[att][i][j]):
                        links.append([value_of_interest,dataframe.iloc[i,int(dataframe[att_pair][i][l][m])]])
            for l in range(len(dataframe[att_triplets][i])):
                for m in range(len(dataframe[att_triplets][i][l])):
                    if int(dataframe[att_triplets][i][l][m]) != int(dataframe[att][i][j]):
                        links.append([value_of_interest,dataframe.iloc[i,int(dataframe[att_triplets][i][l][m])]])
        for j in range(len(dataframe[att_pair][i])):
            for k in range(len(dataframe[att_pair][i][j])):
                value_of_interest = dataframe.iloc[i,int(dataframe[att_pair][i][j][k])]
                for l in range(len(dataframe[att_pair][i])):
                    for m in range(len(dataframe[att_pair][i][l])):
                        if int(dataframe[att_pair][i][l][m]) != int(dataframe[att_pair][i][j][k]):
                            links.append([value_of_interest,dataframe.iloc[i,int(dataframe[att_pair][i][l][m])]])
                for l in range(len(dataframe[att_triplets][i])):
                    for m in range(len(dataframe[att_triplets][i][l])):
                        if int(dataframe[att_triplets][i][l][m]) != int(dataframe[att_pair][i][j][k]):
                            links.append([value_of_interest,dataframe.iloc[i,int(dataframe[att_triplets][i][l][m])]])
        for j in range(len(dataframe[att_triplets][i])):
            for k in range(len(dataframe[att_triplets][i][j])):
                value_of_interest = dataframe.iloc[i,int(dataframe[att_triplets][i][j][k])]
                for l in range(len(dataframe[att_triplets][i])):
                    for m in range(len(dataframe[att_triplets][i][l])):
                        if int(dataframe[att_triplets][i][l][m]) != int(dataframe[att_triplets][i][j][k]):
                            links.append([value_of_interest,dataframe.iloc[i,int(dataframe[att_triplets][i][l][m])]])
    return (links)

def get_nodes(dataframe,method):
    nodes = []
    att = 'single_' + method
    att_pair = 'pairs_' + method
    att_triplets = 'triplets_' + method
    for i in range(len(dataframe)):
        for j in dataframe[att][i]:
            if dataframe.iloc[i,j] not in nodes:
                nodes.append(dataframe.iloc[i,j])
            
        for j in range(len(dataframe[att_pair][i])):
            for k in range(len(dataframe[att_pair][i][j])):
                if dataframe.iloc[i,int(dataframe[att_pair][i][j][k])] not in nodes:
                    nodes.append(dataframe.iloc[i,int(dataframe[att_pair][i][j][k])])
                    
        for j in range(len(dataframe[att_triplets][i])):
            for k in range(len(dataframe[att_triplets][i][j])):
                if dataframe.iloc[i,int(dataframe[att_triplets][i][j][k])] not in nodes:
                    nodes.append(dataframe.iloc[i,int(dataframe[att_triplets][i][j][k])])
                    
    return (nodes)

def build_network(nodes,edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return (G)


def pagerank_score(graph,dataframe):
    pr = pagerank(graph)

    lista = []
    for item in pr.items():
        lista.append(item)
    col1,col2 = [],[]
    for i in lista:
        col1.append(i[0])
        col2.append(i[1])
    df = pd.DataFrame()
    df['key'] = col1
    df['value'] = col2
    df = df.sort_values(by='value',ascending=False).reset_index()
    _tup = []
    for key in df.key:
        j = 0
        while sum(dataframe.iloc[:,j]==key)==0:
            j += 1
        i = 0
        while sum(dataframe.iloc[i,:]==key)==0:
            i += 1
        _tup.append([i,j])
                   
    # initiate the score dictionary 
    score_line = {}
    for i in range(len(dataframe)):
        score_line['{}'.format(i)] = 0

    for i,anomaly in enumerate(_tup):
        score_line['{}'.format(anomaly[0])] += len(_tup)-1
    dataframe['PageRank score'] = score_line.values()
    return (dataframe)

def IsolationForest_detection(dataframe, sample, estimators=100,jobs=-1):
    clf = IsolationForest(n_estimators=estimators,random_state=42,n_jobs=jobs)
    
    dsnodpr = dataframe.copy()
    if len(sample.columns) >= 1:
        predictions = []
        for att_i in range(len(sample.columns)-2):
            train = sample.iloc[:,[att_i,-2]]
            clf.fit(train)
            test = dataframe.iloc[:,[att_i,-2]]
            y_pred_test = clf.predict(test)
            predictions.append(y_pred_test)
        dsnodpr = single_outliers(preds=predictions,dataframe=dsnodpr,method='IF')
    
    if len(sample.columns) >= 2:
        predictions = Isolationforest_pairs(dataframe)
        dsnodpr = outliers_pairs(preds=predictions,dataframe=dsnodpr,method='IF')
    
    if len(sample.columns) >= 3:
        predictions = []
        for att1_i in range(len(dataframe.columns)):
            predections2 = []
            for att2_i in range(att1_i+1,len(dataframe.columns)):
                predictions3 = []
                for att3_i in range(att2_i+1,len(dataframe.columns)):
                    train = sample[[dataframe.columns[att1_i],dataframe.columns[att2_i],dataframe.columns[att3_i]]]
                    clf.fit(train)
                    test = dataframe[[dataframe.columns[att1_i], dataframe.columns[att2_i], dataframe.columns[att3_i]]]
                    y_pred_test = clf.predict(test)
                    predictions3.append(y_pred_test)
                predections2.append(predictions3)
            predictions.append(predections2)
        dsnodpr = outliers_triplets(preds=predictions,dataframe=dsnodpr,method='IF')
    return(dsnodpr)

def autoencoder_detection(dataframe, sample, layers, epo=100, activation_function='relu',
                          batch=32, cont=0.1):
    """
    Description
    -----------
    Builds an autoencoder to detect outliers.
    
    Parameters
    ----------
        dataframe: dataframe to be tested
        sample: a sample of the dataframe to train the model on it
        epo: number of epochs to train the model
        activation_function: Activation functions used in hidden layers
        batch: batch size
        cont: determines the percentage of most anomalous values that is labelled as outliers        
        layers: a list with n elements, n represents the number of layers and the value of each element is the number of neurons in the corresponding layer
        
    Returns:
    -------
        The inputted dataframe with three extra attributes, indidcating which attributes are outliers in each tuple.
        
    """
    autoencoder = pyod.models.auto_encoder.AutoEncoder(hidden_neurons=layers,epochs=epo,batch_size=batch,
                                                     dropout_rate=0.1,hidden_activation=activation_function,
                                                     contamination=cont,verbose=0)
    dsnodpr = dataframe.copy()
    
    if len(sample.columns) >= 1:
        predictions = []
        for att in dataframe.columns:
            #reshaping data
            train = np.array((sample[att]))
            train = train.reshape(-1, 1)
            #training model
            autoencoder.fit(train)
            test = np.array(dataframe[att])
            test = test.reshape((-1,1))
            y_pred_test = autoencoder.predict(test)
            predictions.append(y_pred_test)
        dsnodpr = single_outliers(preds=predictions,dataframe=dsnodpr,method='AE')
    
    if len(sample.columns) >= 2:
        predictions = []
        for i in range(len(sample.columns)):
            predictions2 = []
            for j in range(i+1,len(sample.columns)):
                inp = np.array(sample.iloc[:,i]*sample.iloc[:,j])
                inp = inp.reshape(-1, 1)
                autoencoder.fit(inp)
                test = dataframe.iloc[:,i]*dataframe.iloc[:,j]
                test = np.array((test))
                test = test.reshape(-1, 1)
                predictions2.append(autoencoder.predict(test))
            predictions.append(predictions2)
        dsnodpr = outliers_pairs(preds=predictions,dataframe=dsnodpr,method='AE')
    
    if len(sample.columns) >= 3:
        predictions = []
        for i in range(len(sample.columns)):
            predictions2 = []
            for j in range(i+1,len(sample.columns)):
                predictions3 = []
                for k in range(j+1,len(sample.columns)):
                    inp = np.array(sample.iloc[:,i]*sample.iloc[:,j]*sample.iloc[:,k])
                    inp = inp.reshape(-1, 1)
                    autoencoder.fit(inp)
                    test = dataframe.iloc[:,i]*dataframe.iloc[:,j]*dataframe.iloc[:,k]
                    test = np.array((test))
                    test = test.reshape(-1, 1)
                    predictions3.append(autoencoder.predict(test))
                predictions2.append(predictions3)
            predictions.append(predictions2)
        dsnodpr = outliers_triplets(preds=predictions,dataframe=dsnodpr,method='AE')
    return(dsnodpr)

def loda_detection(dataframe, sample):
    loda = pyod.models.loda.LODA()
    dsnodpr = dataframe.copy()
    
    if len(sample.columns) >= 1:
        predictions = []
        for att in sample.columns:
            train = np.array(sample[att])
            train = train.reshape((-1,1))
            loda.fit(train)
            test = np.array(dataframe[att])
            test = test.reshape((-1,1))
            res = loda.predict(test)
            predictions.append(res)
        dsnodpr = single_outliers(preds=predictions,dataframe=dsnodpr,method='loda')
        
    if len(sample.columns) >= 2:
        predictions = []
        for att1_i in range(len(dataframe.columns)): 
            predections2 = []
            for att2_i in range(att1_i+1,len(sample.columns)):
                train = np.array(sample[[sample.columns[att1_i],sample.columns[att2_i]]])
                train = train.reshape((len(train),2)) 
                loda.fit(train)
                test = np.array(dataframe[[sample.columns[att1_i],sample.columns[att2_i]]])
                test=test.reshape((len(test),2)) 
                y_pred_test = loda.predict(test)
                predections2.append(y_pred_test)
            predictions.append(predections2)
        dsnodpr = outliers_pairs(preds=predictions,dataframe=dsnodpr,method='loda')
        
    if len(sample.columns) >= 3:
        predictions = []
        for att1_i in range(len(sample.columns)): 
            predections2 = []
            for att2_i in range(att1_i+1, len(sample.columns)):
                predictions3 = []
                for att3_i in range(att2_i+1, len(sample.columns)):
                    train=np.array(sample[[sample.columns[att1_i],sample.columns[att2_i],sample.columns[att3_i]]])
                    train=train.reshape((len(train),3)) 
                    loda.fit(train)
                    test=np.array(dataframe[[sample.columns[att1_i],sample.columns[att2_i],sample.columns[att3_i]]])
                    test = test.reshape((len(test),3)) 
                    y_pred_test = loda.predict(test)
                    predictions3.append(y_pred_test)
                predections2.append(predictions3)
            predictions.append(predections2)
        dsnodpr=outliers_triplets(preds=predictions,dataframe=dsnodpr,method='loda')
    return (dsnodpr)


def count_anomalies(dataframe, method):
    at = 'single_' + method
    att_pair = 'pairs_' + method
    att_triplets = 'triplets_' + method
    val,att,counter,tup = [],[],[],[]
    for i in range(len(dataframe[at])):
        for j in range(len(dataframe[at][i])):
            # isolate the anomalous value
            val_1 = dataframe.iloc[i,dataframe[at][i][j]]
            c = 0

            # select the subset with anomalous value
            subdf = dataframe[dataframe.iloc[:,dataframe[at][i][j]]==val_1]

            # count the number this anomalous value takes part in an outlier
            for k in subdf.index:
                for m in subdf[at][k]:
                    if m == dataframe[at][i][j]:
                        c += 1
                for m in subdf[att_pair][k]:
                    if str(dataframe[at][i][j]) in m:
                        c += 1
                for m in subdf[att_triplets][k]:
                    if str(dataframe[at][i][j]) in m:
                        c += 1
            counter.append(c)
            att.append(dataframe[at][i][j])
            val.append(val_1)
            tup.append(i)
    df1 = pd.DataFrame()
    df1['Value'] = val
    df1["Attribute"] = att
    df1["in_Anomalies"] = counter
    df1['tuple'] = tup        

    # ensure that no tuple index is missing
    if len(df1.tuple.unique()) != len(dataframe):
        for i in range(len(dataframe)):
            if i not in df1.tuple.unique():
                df1.loc[len(df1.index)] = [0, 0, 0, i] 
                # if it does append 0 to all columns, so it match the original dataframe

    # count the total anomalies per tuple             
    anomaly_score1 = []
    for i in range(len(dataframe)):
        anomaly_score1.append(sum(df1[df1.tuple == i].in_Anomalies))
        
        
    val,att,counter,tup = [],[],[],[]
    for i in range(len(dataframe[att_pair])):
        for j in range(len(dataframe[att_pair][i])):
            val_1 = dataframe.iloc[i,int(dataframe[att_pair][i][j][0])]
            val_2 = dataframe.iloc[i,int(dataframe[att_pair][i][j][1])]
            list_of_val = [val_1,val_2]
            number_of_anomalies = []
            valuee = []
            for v,value in enumerate(list_of_val):
                c = 0
                subdf = dataframe[dataframe.iloc[:,int(dataframe[att_pair][i][j][v])]==value]

                # count the number this anomalous value takes part in an outlier
                for k in subdf.index:
                    for m in subdf[at][k]:
                        if m == int(dataframe[att_pair][i][j][v]):
                            c += 1
                    for m in subdf[att_pair][k]:
                        if dataframe[att_pair][i][j][v] in m:
                            c += 1
                    for m in subdf[att_triplets][k]:
                        if dataframe[att_pair][i][j][v] in m:
                            c += 1
                number_of_anomalies.append(c)
                valuee.append([value,int(dataframe[att_pair][i][j][v])])
            val.append(valuee[0][0])
            att.append(valuee[0][1])
            counter.append(number_of_anomalies[0])
            val.append(valuee[1][0])
            att.append(valuee[1][1])
            counter.append(number_of_anomalies[1])
            tup.append(i)
            tup.append(i)

    df2 = pd.DataFrame()
    df2['Value'] = val
    df2["Attribute"] = att
    df2["in_Anomalies"] = counter
    df2['tuple'] = tup   

    # ensure that no tuple index is missing
    if len(df2.tuple.unique())!=len(dataframe):
        for i in range(len(dataframe)):
            if i not in df2.tuple.unique():
                df2.loc[len(df2.index)] = [0, 0, 0, i]
    if len(df2.tuple.unique())!=len(dataframe):
        raise Exception("Something didn't work properly")

    # count the total anomalies per tuple 
    anomaly_score2 = []
    for i in range(len(dataframe)):
        anomaly_score2.append(sum(df2[df2.tuple==i].in_Anomalies))
        
    val,att,counter,tup = [],[],[],[]
    for i in range(len(dataframe[att_triplets])):
        for j in range(len(dataframe[att_triplets][i])):
            val_1 = dataframe.iloc[i,int(dataframe[att_triplets][i][j][0])]
            val_2 = dataframe.iloc[i,int(dataframe[att_triplets][i][j][1])]
            val_3 = dataframe.iloc[i,int(dataframe[att_triplets][i][j][2])]
            list_of_val = [val_1,val_2,val_3]
            number_of_anomalies = []
            valuee = []
            for v,value in enumerate(list_of_val):
                c = 0
                subdf = dataframe[dataframe.iloc[:,int(dataframe[att_triplets][i][j][v])]==value]

                # count the number this anomalous value takes part in an outlier
                for k in subdf.index:
                    for m in subdf[at][k]:
                        if m == int(dataframe[att_triplets][i][j][v]):
                            c += 1
                    for m in subdf[att_pair][k]:
                        if dataframe[att_triplets][i][j][v] in m:
                            c += 1
                    for m in subdf[att_triplets][k]:
                        if dataframe[att_triplets][i][j][v] in m:
                            c += 1
                number_of_anomalies.append(c)
                valuee.append([value,int(dataframe[att_triplets][i][j][v])])
            val.append(valuee[0][0])
            att.append(valuee[0][1])
            counter.append(number_of_anomalies[0])
            val.append(valuee[1][0])
            att.append(valuee[1][1])
            counter.append(number_of_anomalies[1])
            tup.append(i)
            tup.append(i)
    df3 = pd.DataFrame()
    df3['Value'] = val
    df3["Attribute"] = att
    df3["in_Anomalies"] = counter
    df3['tuple'] = tup      

    # ensure that no tuple index is missing
    if len(df3.tuple.unique()) != len(dataframe):
        for i in range(len(dataframe)):
            if i not in df3.tuple.unique():
                df3.loc[len(df3.index)] = [0, 0, 0, i]
    if len(df3.tuple.unique()) != len(dataframe):
        raise Exception("Something didn't work properly")

    # count the total anomalies per tuple 
    anomaly_score3 = []
    for i in range(len(dataframe)):
        anomaly_score3.append(sum(df3[df3.tuple==i].in_Anomalies))
        
    rank_score = []
    [rank_score.append(anomaly_score1[i]+anomaly_score2[i]+anomaly_score3[i]) for i in range(len(anomaly_score1))]
    dataframe['Ranking_count'] = rank_score
    return (dataframe)


def ngrams_detection(dataframe,n_g,th):
    copy = dataframe.copy()
    grams = ngrams(dataframe,n_g,1)
    score_line = np.zeros(len(dataframe))
    # single attribute
    for j in range(len(grams)):
        counter = Counter(grams[j])
        ngram_res = pd.DataFrame({'Token':counter.keys(),'Frequency':counter.values()})
        ngram_res.sort_values(by='Frequency',inplace=True)
        ngram_res.reset_index(inplace=True,drop=True)
        avg = np.average(ngram_res.Frequency)
        limit = avg * th
        outliers_grams=[ngram_res.Token[i] for i in range(len(ngram_res.Token)) 
                        if (ngram_res.Frequency[i] < avg-limit or ngram_res.Frequency[i] > avg+limit)]   
        for element in outliers_grams:
            for i in range(len(dataframe)):
                if element in str(dataframe.iloc[i,j]):
                    score_line[i] += 1
    copy["single_ngram"] = score_line

    # pairs of attributes
    score_line = np.zeros(len(dataframe))
    grams = ngrams(dataframe,n_g,2)
    for i in range(len(grams)):
        for j in range(len(grams[i])):
            counter = Counter(grams[i][j])
            ngram_res = pd.DataFrame({'Token':counter.keys(),'Frequency':counter.values()})
            ngram_res.sort_values(by='Frequency',inplace=True)
            ngram_res.reset_index(inplace=True,drop=True)
            avg = np.average(ngram_res.Frequency)
            limit = avg * th
            outliers_grams=[ngram_res.Token[i] for i in range(len(ngram_res.Token)) 
                        if (ngram_res.Frequency[i]<avg-limit or ngram_res.Frequency[i]>avg+limit)]
            for element in outliers_grams:
                for k in range(len(dataframe)):
                    if element in str(dataframe.iloc[k,j+i+1]) or element in str(dataframe.iloc[k,i]):
                        score_line[k] += 1
    copy['pairs_ngram'] = score_line

    score_line = np.zeros(len(dataframe))
    grams = ngrams(dataframe,3,3)
    for i in range(len(grams)):
        for j in range(len(grams[i])):
            for k in range(len(grams[i][j])):
                counter = Counter(grams[i][j][k])
                ngram_res = pd.DataFrame({'Token':counter.keys(),'Frequency':counter.values()})
                ngram_res.sort_values(by='Frequency',inplace=True)
                ngram_res.reset_index(inplace=True,drop=True)
                avg = np.average(ngram_res.Frequency)
                limit = avg*th
                outliers_grams = [ngram_res.Token[i] for i in range(len(ngram_res.Token)) 
                            if (ngram_res.Frequency[i] < avg-limit or ngram_res.Frequency[i] > avg + limit)]
                for element in outliers_grams:
                    for m in range(len(dataframe)):
                        if element in str(dataframe.iloc[m,j+i+1]) or element in str(dataframe.iloc[m,i]) or element in str(dataframe.iloc[m,k+j+2]):
                            score_line[m] += 1
    copy['triplets_ngram']=score_line
    return(copy)

def ngrams_ranking(dataframe,th):
    dataframe['final_score'] = dataframe['triplets_ngram'] + dataframe['pairs_ngram'] + dataframe['single_ngram']
    range_of_val = max(dataframe['final_score']) - min(dataframe['final_score'])
    limit = (1-th) * range_of_val + min(dataframe['final_score'])
    return(dataframe[dataframe['final_score']>limit].index)

def dataframe_score(dataframe,norm_df,method,function, th=0.1):
    # ranking of attributes
    att='single_' + method
    att_pair='pairs_' + method
    att_triplets='triplets_' + method
    att_anomalies=[]
    for i in range(len(norm_df.columns)): 
        c = 0
        for j in range(len(dataframe)):
            if i in dataframe[att][j]:
                c += 1
            for pair in dataframe[att_pair][j]:
                if str(i) in pair:
                    c += 1
            for triplet in dataframe[att_triplets][j]:
                if str(i) in triplet:
                    c += 1
        att_anomalies.append(c)
    output = pd.DataFrame({'Attribute':norm_df.columns,'Anomalies':att_anomalies})
    norm_factor = len(norm_df)*(1+(len(norm_df.columns)-1)**2)
    output.Anomalies = output.Anomalies/norm_factor
    if function=='mean':
        score = np.mean(output.Anomalies)
    elif function=='median':
        score = np.median(output.Anomalies)
    else:
        raise Exception('Invalid function to calculate the score of the Dataset')
    return [output[output.Anomalies>1-th].index, score]
def maximum_score_att(dataframe,n_g):
    maximum_score=np.zeros(len(dataframe.columns))
    maximum_score_pairs=np.zeros(len(dataframe.columns))
    maximum_score_triplets=np.zeros(len(dataframe.columns))
    
    if len(dataframe.columns)>=1:

        for j in range(len(dataframe.columns)):
            c=0
            for i in range(len(dataframe[dataframe.columns[j]])):
                maximum_score[j] += (len(str(dataframe[dataframe.columns[j]][i]))- n_g +1) 
        
        
    # pairs 
    if len(dataframe.columns)>=2:

        for j in range(len(dataframe.columns)):
            for k in range(len(dataframe.columns)):
                if dataframe.columns[j]!=dataframe.columns[k]:
                    for i in range(len(dataframe)):
                        maximum_score_pairs[j] += (len(str(dataframe[dataframe.columns[j]][i]) + str(dataframe[dataframe.columns[k]][i]))-n_g+1)
                        maximum_score_pairs[k] += (len(str(dataframe[dataframe.columns[j]][i]) + str(dataframe[dataframe.columns[k]][i]))-n_g+1)
    # triplets 
    if len(dataframe.columns) >= 3:

        for j in range(len(dataframe.columns)):
            for k in range(len(dataframe.columns)):
                for m in range(len(dataframe.columns)):
                    if dataframe.columns[j] != dataframe.columns[k] and dataframe.columns[j]!=dataframe.columns[m] and dataframe.columns[m] != dataframe.columns[k]:
                        for i in range(len(dataframe)):
                            maximum_score_triplets[j] += (len(str(dataframe[dataframe.columns[j]][i]) + str(dataframe[dataframe.columns[m]][i]) + str(dataframe[dataframe.columns[k]][i])) - n_g + 1)
                            maximum_score_triplets[k] += (len(str(dataframe[dataframe.columns[j]][i]) + str(dataframe[dataframe.columns[m]][i]) + str(dataframe[dataframe.columns[k]][i])) - n_g + 1)
                            maximum_score_triplets[m] += (len(str(dataframe[dataframe.columns[j]][i]) + str(dataframe[dataframe.columns[m]][i]) + str(dataframe[dataframe.columns[k]][i])) - n_g + 1)
    return (maximum_score+maximum_score_pairs+maximum_score_triplets)

def ngram_score_att(dataframe, n_g, threshold = 0.1):
    single_score = np.zeros(len(dataframe.columns))
    pairs_score = np.zeros(len(dataframe.columns))
    triplet_score = np.zeros(len(dataframe.columns))
    if len(dataframe.columns) >= 1:
        grams = ngrams(dataframe, n_g, 1)
        for i in range(len(grams)):
            counter = Counter(grams[i])
            ngram_res = pd.DataFrame({'Token':counter.keys(),'Frequency':counter.values()})
            ngram_res.sort_values(by='Frequency',inplace=True)
            ngram_res.reset_index(inplace=True,drop=True)
            avg = np.average(ngram_res.Frequency)
            limit = avg*threshold
            outliers_grams = [ngram_res.Token[i] for i in range(len(ngram_res.Token)) 
                        if (ngram_res.Frequency[i]<avg-limit or ngram_res.Frequency[i]>avg+limit)]
            for gram in outliers_grams:
                for m in range(len(dataframe)):
                    if gram in str(dataframe.iloc[m,i]):
                        single_score[i] += 1
    if len(dataframe.columns) >= 2:
        grams=ngrams(dataframe, n_g, 2)
        for i in range(len(grams)):
            for j in range(len(grams[i])):
                counter = Counter(grams[i][j])
                ngram_res = pd.DataFrame({'Token':counter.keys(),'Frequency':counter.values()})
                ngram_res.sort_values(by='Frequency',inplace=True)
                ngram_res.reset_index(inplace=True,drop=True)
                avg=np.average(ngram_res.Frequency)
                limit = avg*threshold
                outliers_grams = [ngram_res.Token[i] for i in range(len(ngram_res.Token)) 
                            if (ngram_res.Frequency[i]<avg-limit or ngram_res.Frequency[i]>avg+limit)]
                for gram in outliers_grams:
                    for m in range(len(dataframe)):
                        if gram in str(dataframe.iloc[m,i])+str(dataframe.iloc[m,j+i+1]):
                            pairs_score[i] += 1
                            pairs_score[j+i+1] += 1
    if len(dataframe.columns) >= 3:
        grams=ngrams(dataframe, n_g, 3)
        for i in range(len(grams)):
            for j in range(len(grams[i])):
                for k in range(len(grams[i][j])):
                    counter = Counter(grams[i][j][k])
                    ngram_res = pd.DataFrame({'Token':counter.keys(),'Frequency':counter.values()})
                    ngram_res.sort_values(by='Frequency',inplace=True)
                    ngram_res.reset_index(inplace=True,drop=True)
                    avg = np.average(ngram_res.Frequency)
                    limit = avg*threshold
                    outliers_grams = [ngram_res.Token[i] for i in range(len(ngram_res.Token)) 
                                if (ngram_res.Frequency[i] < avg-limit or ngram_res.Frequency[i]>avg+limit)]
                    for gram in outliers_grams:
                        for m in range(len(dataframe)):
                            if gram in str(dataframe.iloc[m,i]) + str(dataframe.iloc[m,j+i+1]) + str(dataframe.iloc[m,k+j+i+2]):
                                triplet_score[i] += 1
                                triplet_score[j+i+1] += 1
                                triplet_score[k+j+i+2] += 1
    return (single_score+pairs_score+triplet_score)

def read_config():
    config = configparser.ConfigParser()
    config.read(r"C:\Users\milia\thesis\config\configuration.ini")
    return config