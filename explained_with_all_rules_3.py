
"""
In this version is clean version for version 2. Here we first made the subsamples and then trained. In the subsamples there is no dataset. Only sampling of the features.

"""
import os
import sys
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pandas as pd
import sigdirect
from random import seed
from random import random
from random import randrange
import copy

class _Preprocess:

    def __init__(self):
        self._label_encoder = None

    def preprocess_data(self, raw_data):
        """ Given one of UCI files specific to SigDirect paper data,
            transform it to the common form used in sklearn datasets"""
        transaction_data = [list(map(int, x.strip().split())) for x in raw_data]
        max_val = max([max(x[:-1]) for x in transaction_data])
        X, y = [], []

        for transaction in transaction_data:
            positions = np.array(transaction[:-1]) - 1
            transaction_np = np.zeros((max_val))
            transaction_np[positions] = 1
            X.append(transaction_np)
            y.append(transaction[-1])
        X = np.array(X)
        y = np.array(y)

        # converting labels
        if self._label_encoder is None:  # train time
            unique_classes = np.unique(y)
            #print(unique_classes)
            self._label_encoder = defaultdict(lambda: 0, zip(unique_classes, range(len(unique_classes))))
            y = np.vectorize(self._label_encoder.get)(y)
        else:  # test time
            y = np.vectorize(lambda a: self._label_encoder[a])(y)

        return X, y


def test_uci():

    data=["flare","pima","glass","heart","hepati","wine","anneal","horse","ionosphere","adult"]
    #data=["hepati"]
    #data=["horse"]
   
    for dataset_name in data:
        t1=time.time()
        f=open("savedrules.txt", "w")
        f.close()
        memory=0
        start_time = time.time()
        #print(ratio, end=" ")
        print(dataset_name,end=" ")
        
        prep = _Preprocess()

        # load the training data and pre-process it
        train_filename = os.path.join('uci', '{}.txt'.format(dataset_name))
        with open(train_filename) as f:
            raw_data = f.read().strip().split('\n')

        X, y = prep.preprocess_data(raw_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        temp_x=list(X_train)
        temp_y=list(y_train)
        n=np.unique(y)
        min_count=sys.maxsize
        for i in n:
            if(np.count_nonzero(y == i)<min_count):
                min_count=np.count_nonzero(y == i)
                #print(i,np.count_nonzero(y == i) )
        #for i in y:
        #    print(i, end=" ")

        #print("\nminimum",min_count)
        boost_X=[]
        boost_y=[]

        for i in range(100):
            try:
                #print(i)
                clf = sigdirect.SigDirect(get_logs=sys.stdout)
                random.seed(1)


                sample, sample_test,temp = subsample(X_train, X_test,.6)
                temp2=[]
                if(isinstance(temp,list)):
                    #print(temp)
                    #print("features for base learner",base_l,"\ntemp2",end=" ")

                    for i in temp:
                        if(X_test[0][i]==1):
                            #print(i, end=",")
                            temp2.append(i)
                #print(type(sample))
                if(not isinstance(sample, pd.core.frame.DataFrame)):
                    break
                temp_sample=sample
                sample = np.array(sample)
                temp_sample=sample
                sample_test = np.array(sample_test)
                #print(sample.shape)
                # print("sample size",len(sample))
                # sample_y = np.array(sample_y)
                g, f,m = clf.fit(sample, y_train,temp,temp2)
                #########################################################
                ######################################################
                
                pred=clf.predict(sample,2,temp)
                sample = []
                test_sample = pd.DataFrame()
                y_sample = []
                for k in range(len(pred)):
                    #print(y_train[k], pred[k])
                    if(y_train[k]!=pred[k]):
                       
                        sample.append(temp_sample[k])
                        
                        y_sample.append(y_train[k])
                clf = sigdirect.SigDirect(get_logs=sys.stdout)
                sample = np.array(sample)
                y_train=np.array(y_train)

                g, f,m = clf.fit(np.array(sample), np.array(y_sample),temp,temp2)
                
                ##############################################################
                ################################################################
                #print(f)

            except:
                pass
       
        #clf=sigdirect.SigDirect(get_logs=sys.stdout)
        X_train=np.array(X_train)
        X_test=np.array(X_test)
        #clf.get_final_rules(X_train,y_train,temp,temp2)

        ######################
        
        #for r in clf._final_rules:
        #    print(r)


        ########################



        #pred=clf.predict(X,2,temp)
        #acc=-1
        #print("accuracy_score: ", accuracy_score(y_train, pred))
        #####################################
        #boosting step
        """
        while(acc< accuracy_score(y, pred)):
            print("accuracy_score: ", accuracy_score(y, pred))
            acc= accuracy_score(y, pred)
            temp_x=list(X)
            temp_y=list(y)
            boost_X=[]
            boost_y=[]
            for i in range(len(X)):
                if(y[i]!=pred[i]):
                    
                    boost_X.append(X[i])
                    boost_y.append(y[i])
            print("misclassified",len(boost_y))
            boost_X=np.array(boost_X)
            boost_y=np.array(boost_y)
            clf=sigdirect.SigDirect(get_logs=sys.stdout)
            clf.fit(boost_X,boost_y)
            clf.fit(boost_X,boost_y)
            ######################################
            clf=sigdirect.SigDirect(get_logs=sys.stdout)
            clf.get_final_rules(X,y)
            pred=clf.predict(X)
        """
        
        clf=sigdirect.SigDirect(get_logs=sys.stdout)
        clf.get_final_rules(X_train,y_train,temp,temp2)
        pred=clf.predict(X_test,2,temp)
        print("accuracy: ", accuracy_score(y_test, pred), "precision:",precision_score(y_test, pred, average='macro'), recall_score(y_test, pred, average='macro'), "time:" ,time.time()-t1)
        """
        print("y_test", end=" ")
        for i in y_test:
            print(i,end=" ")
        print("\npred__", end=" ")
        for i in pred:
            print(i,end=" ")
        print()
        """
        #print("accuracy_score: ", accuracy_score(y_test, pred))



# Create a random subsample from the dataset with replacement
import random


collect=[]
taken=[]
def subsample(dataset, test_dataset,ov):
    df = pd.DataFrame(dataset)
    df2 = pd.DataFrame(test_dataset)
    feat=20 #heart 20
    run_loop=True
    random.seed()
    tracker=0 #count how many times subsampling done


    sample = pd.DataFrame()
    test_sample = pd.DataFrame()
    y_sample = pd.DataFrame()
    index = []
    #n_feature = df.shape[1]
    #print("shape",df.shape[1])
    n_feature = feat


    #print("nfeature",n_feature)
    temp = []
    count=0
    random.seed()
    while (len(temp) < feat and len(temp)<df.shape[1]-1):
        index = random.randint(0, df.shape[1] - 1)
        temp.append(index)
        sample[sample.shape] = df[index]
        test_sample[test_sample.shape] = df2[index]
            
    return sample, test_sample, temp

    


if __name__ == '__main__':
    test_uci()