#!/usr/bin/env python3

import os
import sys
import sys
import time
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix


import sigdirect
import dataset_transformation

class _Preprocess:

    def __init__(self):
        self._label_encoder = None

    def preprocess_data(self, raw_data):
        """ Given one of UCI files specific to SigDirect paper data,
            transform it to the common form used in sklearn datasets"""
        transaction_data =  [list(map(int, x.strip().split())) for x in raw_data]
        max_val = max([max(x[:-1]) for x in transaction_data])
        X,y = [], []

        for transaction in transaction_data:
            positions = np.array(transaction[:-1]) - 1
            transaction_np = np.zeros((max_val))
            transaction_np[positions] = 1
            X.append(transaction_np)
            y.append(transaction[-1])
        X = np.array(X)
        y = np.array(y)

        # converting labels
        if self._label_encoder is None:# train time
            unique_classes = np.unique(y)
            self._label_encoder = defaultdict(lambda: 0, zip(unique_classes, range(len(unique_classes))))
            y = np.vectorize(self._label_encoder.get)(y)
        else:# test time
            y = np.vectorize(lambda a: self._label_encoder[a])(y)

        return X,y

def prep_data(dataset_name,counter):
    import pandas as pd
    import warnings
    import pandas as pd
    import random
    from random import randint
    from sklearn.model_selection import train_test_split
    warnings.filterwarnings('ignore')
    
    input_name  = 'uci/' + dataset_name + '.txt' #train file
    
    #nameFile="datasets_new_originalfiles" +"\\"+fileName+".names"
    
    #method 1 for readin using pandas dataframe
    sep = ' '
    
    with open(input_name, 'r') as f:
        data = f.read().strip().split('\n')
    dataset = [line.strip().split(sep) for line in data]        
    
   
    df=pd.DataFrame(dataset)
    #print("df:",df )
    dforiginal=df
    #masking all the nones if any
    mask = df.applymap(lambda x: x is None)
    cols = df.columns[(mask).any()]
    for col in df[cols]:
        df.loc[mask[col], col] = ''
    dforiginal=df    
     
    dfnew=df
    #X = df.iloc[:,:-1].values
    #this line replaces all empty spaces with nan. this is done to get the last col values.
    dfnew = dfnew.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace('', np.nan)
    
    #now u can get the last col values that is the labels.
    Y=dfnew.groupby(['label'] * dfnew.shape[1], 1).agg('last')
    
    dfnew=dfnew.where(pd.notnull(dfnew), None)
    #dfnew = dfnew.apply(lambda x: x.str.strip() if isinstance(x, str) else x).replace(np.nan,None)
    dfnew=dfnew.values.tolist()
    
    #print()
    for k in range(len(dfnew)):
        dfnew[k]=[x for x in dfnew[k] if x is not None]        #removing none from list
    
   
    
    #print()
    for i in dfnew:
        i.pop()
    X=dfnew    
    
   
    #remove none and then then pop out the last element
    #masking all the nones if any
    
    
    #Y = df.iloc[:,-1].values # Here first : means fetch all rows :-1 means except last column
    #X = df.iloc[:,:-1].values # : is fetch all rows 3 means 3rd column  
    
    #print("len(X)",len(X))
    #print("len(Y)",len(Y))
    
    #X=np.array([np.array(xi) for xi in X])
    #X=np.asarray(X) 
    #print("Now x: ",X)
    
    X=pd.DataFrame(X)
    
    mask = X.applymap(lambda x: x is None)
    cols = X.columns[(mask).any()]
    for col in X[cols]:
        X.loc[mask[col], col] = '' 
   
    
    #print("X: ",X)
    #print("Y: ",Y)
    #print("Len of X: ",X.shape)
    #print("Len of Y: ",Y.shape)
  
    #now split the prune set
    #use smote
    # did smote here, then discretization on weka, but you can change that and do it inoython as well. and then tokenize it via data transformation code and then
    #use any sig classification model.
    '''
    print(len(Y))
    from imblearn.over_sampling import SMOTE
    smt = SMOTE(k_neighbors=5, random_state=1206)
    X, Y = smt.fit_sample(X, Y)
    print(len(Y))    
    dataset_transformation
    print(stopp)
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.10, shuffle='true')
    
    
         
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, shuffle='true')
    #X_val=X_train
    #y_val=y_train
    #print(X_train)
    #print(y_train)
    '''
    print("###########################len(X_train): ",len(X_train))
    
    print("len(X_test): ",len(X_test))
    print("len(y_train): ",len(y_train))
    print("len(y_test): ",len(y_test))
    print("len(X_val): ",len(X_val))
    print("len(y_val): ",len(y_val))
    '''
    X_test_original=X_test
    #df3_test=pd.DataFrame(list(X_test_original))              
    df_Xtest=X_test_original              
    #test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_testoriginal'+ str(num_run) +'.txt'
    #df_Xtest.to_csv(test_name,sep=' ',index=False,header=False)  
    
    df3_test=y_test 
    #test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_ytestoriginal'+ str(num_run) +'.txt'
    #df3_test.to_csv(test_name,sep=' ',index=False,header=False)          
            
    df_Xtrain=X_train                     
    #test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_trainoriginal'+ str(num_run) +'.txt'
    #df_Xtrain.to_csv(test_name,sep=' ',index=False,header=False)
    
    df3_test=y_train                     
    #test_name  = 'datasets_new_ensemble_boosting/' + dataset_name + '_ytrainoriginal'+ str(num_run) +'.txt'
    #df3_test.to_csv(test_name,sep=' ',index=False,header=False)
    
    #you should make new datasets.
    #df1=pd.DataFrame(list(X_train))
    #df2=pd.DataFrame(list(y_train)) 
    #print(X_train)
    #print(y_train)
    df1=pd.DataFrame(X_train)
    df2=pd.DataFrame(y_train)   
    
    #for val set
    df1_val=pd.DataFrame(X_val)
    df2_val=pd.DataFrame(y_val)
    Xval_subsample = pd.concat([df1_val,df2_val], axis=1)#this is xval
    Xval_subsample=Xval_subsample.reset_index(drop=True)
    Xval_subsample.columns = list(range(0, X_train.shape[1]+1))        
    #print("Xval_subsample: ",Xval_subsample)       
    #print("NOW??????????????????df1: ",df1)
    
    
    df4y_test=pd.DataFrame(X_test) 
    y_test=pd.DataFrame(y_test)
    Xtest_subsample = pd.concat([df4y_test,y_test], axis=1) #this is xtest
    Xtest_subsample=Xtest_subsample.reset_index(drop=True)
    Xtest_subsample.columns = list(range(0, X_train.shape[1]+1))
    
    Xtotal_subsample = pd.concat([df1,df2], axis=1) #this is xtotal
    
    Xtotal_subsample=Xtotal_subsample.reset_index(drop=True)
    Xtotal_subsample.columns = list(range(0, X_train.shape[1]+1))
    
    test_name  = 'datasets_new/' + dataset_name + '_validation'+ str(counter) +'.txt'    
    Xval_subsample.to_csv(test_name,sep=' ',index=False,header=False)  
    
    test_name  = 'datasets_new/' + dataset_name + '_train'+ str(counter) +'.txt'
    Xtotal_subsample.to_csv(test_name,sep=' ',index=False,header=False)
    
    test_name  = 'datasets_new/' + dataset_name + '_test'+ str(counter) +'.txt'
    Xtest_subsample.to_csv(test_name,sep=' ',index=False,header=False)    
    
def test_uci():

    data=[ "pima","glass","pageBlocks","hepati","wine","anneal","horse","ionosphere","adult"]
    data=["pima"]
    for dataset_name in data:
        #print(dataset_name)
        start_index = 1
        
        start_index=1
        final_index = 5
        k = final_index - start_index + 1

        all_pred_y = defaultdict(list)
        all_true_y = []

        # counting number of rules before and after pruning
        generated_counter = 0
        final_counter     = 0
        avg = [0.0] * 4
        f1li=[]
        precisionli=[]
        recallli=[]
        accuracyli=[]
        
        memory=-1
        for index in range(start_index, final_index +1):
            tt1 = time.time()
            counter=index
            prep_data(dataset_name,counter)
            #print(stopp)        
            #print(index)
            prep = _Preprocess()
            
            #################################33
            #this is for val file
            
            # load the training data and pre-process it
            #train_filename  =   'datasets_bag_rc/' + dataset_name + '_val'+ str(counter) +'.txt'
            train_filename  =   'datasets_new/' + dataset_name + '_validation'+ str(counter) +'.txt'
                    
            #train_filename = os.path.join('uci', '{}_tr{}.txt'.format(dataset_name, index))
            with open(train_filename) as f:
                raw_data = f.read().strip().split('\n')
            X_val,y_val = prep.preprocess_data(raw_data)        
            
            
            #################################33

            # load the training data and pre-process it
            #train_filename  =   'datasets_bag_rc/' + dataset_name + '_train'+ str(counter) +'.txt'
            train_filename  =   'datasets_new/' + dataset_name + '_train'+ str(counter) +'.txt'
                    
            #train_filename = os.path.join('uci', '{}_tr{}.txt'.format(dataset_name, index))
            with open(train_filename) as f:
                raw_data = f.read().strip().split('\n')
            X,y = prep.preprocess_data(raw_data)
            clf = sigdirect.SigDirect(get_logs=sys.stdout)
            generated_c, final_c,m = clf.fit(X, y)
            if(m>memory):
                memory=m
            generated_counter += generated_c
            final_counter     += final_c

            # load the test data and pre-process it.
            #test_filename  =   'datasets_bag_rc/' + dataset_name + '_test'+ str(counter) +'.txt'
            test_filename  =   'datasets_new/' + dataset_name + '_test'+ str(counter) +'.txt'
            
            #test_filename  = os.path.join('uci', '{}_ts{}.txt'.format(dataset_name, index))
            with open(test_filename) as f:
                raw_data = f.read().strip().split('\n')
            X,y = prep.preprocess_data(raw_data)
            y_test=y
            # evaluate the classifier using different heuristics for pruning
            
            for hrs in (1,2,3):
                y_pred = clf.predict(X, hrs)
                #print('ACC '.format(hrs), accuracy_score(y, y_pred))
                #print("{:.4f}".format(accuracy_score(y, y_pred), end=" ")
                avg[hrs] += accuracy_score(y, y_pred)

                all_pred_y[hrs].extend(y_pred)

            all_true_y.extend(list(y))
            tt2 = time.time()
            ##############
            '''
        
            # accuracy: (tp + tn) / (p + n)
            accuracy = accuracy_score(y_test, y_pred)
            accuracyli.append(accuracy)
            print('Accuracy: %f' % accuracy)
            # precision tp / (tp + fp)
            precision = precision_score(y_test, y_pred, average="weighted")
            precisionli.append(precision)
            print('Precision: %f' % precision)
            # recall: tp / (tp + fn)
            recall = recall_score(y_test, y_pred, average="weighted")
            recallli.append(recall)
            print('Recall: %f' % recall)
            # f1: 2 tp / (2 tp + fp + fn)
            f1 = f1_score(y_test,y_pred, average="weighted" )
            f1li.append(f1)
            print('F1 score: %f' % f1)

            ###############'''
            #print('\n\n')

        print(dataset_name, end=" ")
        for hrs in (1,2,3):
            print("{:.4f}".format(accuracy_score(all_true_y, all_pred_y[hrs])), end=" ")
        print(memory,"{:.2f}".format(time.time() - tt1))
        #print('TOTAL TIME:', time.time()-tt1)
        #print("y_pred: ",y_pred)
        #print("y_test: ",y_test)
        #print("values of  k is: ",k)
        #print('accuracyli avg: ',sum(accuracyli)/k)
        #print('precisionli avg: ',sum(precisionli)/k)
        #print('recallli avg: ',sum(recallli)/k)
        #print('f1li avg: ',sum(f1li)/k)
    
    #return y_pred,y_test
    

if __name__ == '__main__':
    test_uci()
#if __name__ == '__main__':
'''
def main(dataset_name,counter):
    predictions,y_test=test_uci(dataset_name,counter)
    print("***************************************************")
    print("predictions: ",predictions)
    print("y_test: ",y_test)
    print("***************************************************")
    return predictions,y_test
    '''
