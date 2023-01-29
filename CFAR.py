
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

    data=["pima","glass","heart","hepati","wine","anneal","horse","adult","ionosphere"]
    data=["zoo", "pageblocks"]
    T=[.1,.2,.3,.4,.5,.6,.7,.8,.9,1]
    #data=["hepati"]

    for dataset_name in data:


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
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        temp_x=list(X_train)
        temp_y=list(y_train)
        n=np.unique(y)
        min_count=sys.maxsize
        for i in n:
            if(np.count_nonzero(y == i)<min_count):
                min_count=np.count_nonzero(y == i)


        for i in range(100):
            try:
                #print(i)
                clf = sigdirect.SigDirect(get_logs=sys.stdout)
                random.seed(time.time())


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

                sample = np.array(sample)
                sample_test = np.array(sample_test)
                #print(sample.shape)
                # print("sample size",len(sample))
                # sample_y = np.array(sample_y)
                g, f,m = clf.fit(sample, y_train,temp,temp2)
                #print(f)

            except:
                pass
        acc=[]
        for th in T:
            #print("Th value",th, end=" ")          
            #clf=sigdirect.SigDirect(get_logs=sys.stdout)
            X_train=np.array(X_train)
            X_test=np.array(X_test)
          
            clf=sigdirect.SigDirect(get_logs=sys.stdout)

            clf.get_final_rules(X_train,y_train,temp,temp2,th,False)


            pred=clf.predict(X_test,2,temp)
            #print("accuracy: ", accuracy_score(y_test, pred), "precision:",precision_score(y_test, pred, average='macro'), recall_score(y_test, pred, average='macro'))
            print(th,"accuracy: ", accuracy_score(y_test, pred))
            acc.append(accuracy_score(y_test, pred))
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
        print(dataset_name,acc)
        ind=acc.index(max(acc))
        th=T[ind]
        #print(acc)
        #print(th)
        print("max accuracy at:",th)
        clf=sigdirect.SigDirect(get_logs=sys.stdout)
        clf.get_final_rules(X_train,y_train,temp,temp2,th,True)
        pred=clf.predict(X_val,2,temp)
        print("accuracy: ", accuracy_score(y_val, pred), "precision:",precision_score(y_val, pred, average='macro'), recall_score(y_val, pred, average='macro'))




# Create a random subsample from the dataset with replacement
import random


collect=[]
taken=[]
def subsample(dataset, test_dataset,ov):
    df = pd.DataFrame(dataset)
    df2 = pd.DataFrame(test_dataset)
    feat=30
    run_loop=True
    random.seed(time.time())
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
    random.seed(time.time())
    while (len(temp) < feat and len(temp)<df.shape[1]-1):
        index = random.randint(0, df.shape[1] - 1)
        temp.append(index)
        sample[sample.shape] = df[index]
        test_sample[test_sample.shape] = df2[index]
            
    return sample, test_sample, temp

    


if __name__ == '__main__':
    test_uci()