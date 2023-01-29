
"""
This version changed the previous version by changing the percentage to fixed number of features

"""
import os
import sys
import time
from collections import defaultdict
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import sigdirect
from random import seed
from random import random
from random import randrange


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
            self._label_encoder = defaultdict(lambda: 0, zip(unique_classes, range(len(unique_classes))))
            y = np.vectorize(self._label_encoder.get)(y)
        else:  # test time
            y = np.vectorize(lambda a: self._label_encoder[a])(y)

        return X, y


def test_uci():

    #orig_stdout = sys.stdout
    #file = open('out.txt', 'w')
    #sys.stdout = file
    data=["zoo"]
    data=["zoo", "pima","glass","pageBlocks","heart","hepati","wine","anneal","horse","ionosphere","adult"]
    data=["zoo"]
    for ov in[18]: #[3,6,9,12,15,18,21,24,27,30]:
        print("\n\nwith",ov/30,"overlap")

        for dataset_name in data:


            memory=0
            start_time = time.time()
            #print(ratio, end=" ")
            print(dataset_name, end=" ")
            all_pred_y = defaultdict(list)
            all_true_y = []
            collect.clear()
            taken.clear()
            # counting number of rules before and after pruning
            generated_counter = 0
            final_counter = 0
            avg = [0.0] * 4

            tt1 = time.time()
            predictors = []
            # print(index)
            prep = _Preprocess()

            # load the training data and pre-process it
            train_filename = os.path.join('uci', '{}.txt'.format(dataset_name))
            with open(train_filename) as f:
                raw_data = f.read().strip().split('\n')
            
            X, y = prep.preprocess_data(raw_data)
            #print(X.shape)
            #X=X[:300,:]
            #y=y[:300]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            generated_c = 0
            final_c = 0
            """
            clf = sigdirect.SigDirect(get_logs=sys.stdout)
            g, f = clf.fit(X_train, y_train)
            for i in (1, 2, 3):
                y_pred=clf.predict(X_test, i)
                print('ACC S{}:'.format(i), accuracy_score(y_test, y_pred))
            """
            pred1 = []
            pred2 = []
            pred3 = []
            no_of_predictor=0
            #print("hello")
            indexes = [i for i, j in enumerate(X_test[3]) if j == 1]
            print("test",indexes)
            print("vector",X_test[3])
            for base_l in range(100):
                #print(base_l)
                print("\n")
                clf = sigdirect.SigDirect(get_logs=sys.stdout)
                seed(1)
                # True mean
                #dataset = [[randrange(len(X_train))] for i in range(20)]


                sample, sample_test,temp = subsample(X_train, X_test,ov)
                print("features for base learner",temp)
                temp2=[]
                if(isinstance(temp,list)):
                    #print(temp)
                    #print("features for base learner",base_l,"\ntemp2",end=" ")

                    for i in temp:
                        if(X_test[0][i]==1):
                            #print(i, end=",")
                            temp2.append(i)

                    print()

                if(not isinstance(sample, pd.core.frame.DataFrame)):
                    break
                sample = np.array(sample)
                sample_test = np.array(sample_test)
                #print(sample.shape)
                # print("sample size",len(sample))
                # sample_y = np.array(sample_y)
                no_of_predictor+=1
                g, f,m = clf.fit(sample, y_train,temp,temp2)
                if(m>memory):
                    memory=m
                #print(memory)
                #pred1.append(clf.predict(sample_test, 1))
                #print(pred1)
                sample_test=sample_test[3:4]
                pred2.append(clf.predict(sample_test, 2,temp))
                #pred3.append(clf.predict(sample_test, 3))

                generated_c += g
                final_c += f

            generated_counter = generated_c / no_of_predictor
            final_counter = final_c / no_of_predictor


            for i in pred2:
                print(i)
            #print(X_test)
            final_prediction = []
            pred2 = np.array(pred2)
            pred2 = pred2.transpose()
            for i in pred2:
                i = list(i)
                final_prediction.append(max(i, key=i.count))
            y_test=y_test[3:4]
            print(y_test,final_prediction)
            print( "accuracy", accuracy_score(y_test, final_prediction), end=" " )


         


            end_time = time.time()
            print( "time",end_time - start_time,"memory",  memory, "estimator", base_l)
            #print("base learner",count_base_learner(X.shape[1]))

            """
            # load the test data and pre-process it.
            test_filename = os.path.join('uci', '{}_ts{}.txt'.format(dataset_name, index))
            with open(test_filename) as f:
                raw_data = f.read().strip().split('\n')
            X, y = prep.preprocess_data(raw_data)
            # evaluate the classifier using different heuristics for pruning
            for hrs in (1,2,3):
                y_pred = clf.predict(X, hrs)
                print('ACC S{}:'.format(hrs), accuracy_score(y, y_pred))
                avg[hrs] += accuracy_score(y, y_pred)
                all_pred_y[hrs].extend(y_pred)
            """
            # all_true_y.extend(list(y))
            # print('\n\n')

            """
            print("final score")
            print("=================")
            print("accuracy",accuracy_score(y, final_prediction))
            print(dataset_name)
            for hrs in (1,2,3):
                print('AVG ACC S{}:'.format(hrs), accuracy_score(all_true_y, all_pred_y[hrs]))
            print('INITIAL RULES: {} ---- FINAL RULES: {}'.format(generated_counter/k, final_counter/k))
            print('TOTAL TIME:', time.time()-tt1)
            """

            
    #sys.stdout = orig_stdout



# Create a random subsample from the dataset with replacement
import random
collect=[]
taken=[]
def subsample(dataset, test_dataset,ov):
    df = pd.DataFrame(dataset)
    df2 = pd.DataFrame(test_dataset)
    feat=30
    run_loop=True
    random.seed()
    tracker=0 #count how many times subsampling done
    if(len(np.unique( np.array(taken)))==df.shape[1]):
        return -1,-1,-1

    while(run_loop and tracker<100):
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
        while (len(temp) < n_feature and len(temp)<df.shape[1]-1):
            index = random.randint(0, df.shape[1] - 1)
            temp.append(index)
            # print(index)
            sample[sample.shape] = df[index]
            test_sample[test_sample.shape] = df2[index]
                
        if(len(collect)==0):
            break  
        all_count=[]           
        for c in collect:
            count=0           
            for i in temp:
                if(i in c):
                    count+=1
            #print(count, n_sample,int(overlap))
            all_count.append(count)
        #print(all_count)   
        c2 = [i for i in all_count if i >ov]

        if(len(c2)>0):
            tracker+=1
        else:
            break

    if(tracker<100):
        collect.append(list(temp))
        taken.extend(temp)
        return sample, test_sample, temp
    else:    
        return -1,-1,-1
    




if __name__ == '__main__':
    test_uci()