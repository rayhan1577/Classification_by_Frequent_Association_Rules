import os
import sys
import time
from collections import defaultdict
from sklearn.metrics import accuracy_score, classification_report,precision_score,recall_score,f1_score
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import sigdirect


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



    #dataset=["iris","breast","glass","heart","hepati","wine","pima","zoo","flare","led7","pageBlocks","anneal","horse"]
    dataset=["flare"]
    #dataset=["pima","glass","pageblocks","heart","hepati","wine","anneal","horse","ionosphere","mushroom","adult"]
    for dataset_name in dataset:
        t1=time.time()
        print(dataset_name, end=" ")
        # counting number of rules before and after pruning
        generated_counter = 0
        final_counter = 0
        avg = [0.0] * 4

        tt1 = time.time()

        prep = _Preprocess()

        # load the training data and pre-process it
        train_filename = os.path.join('uci', '{}.txt'.format(dataset_name))
        with open(train_filename) as f:
            raw_data = f.read().strip().split('\n')
        X, y = prep.preprocess_data(raw_data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = sigdirect.SigDirect(get_logs=sys.stdout)
        generated_counter, final_counter,memory = clf.fit(X_train, y_train,[],[])
        

        final_prediction = clf.predict(X_test, 2) 
        print(y_test)
        print(final_prediction)
        print("Accuracy:", accuracy_score(y_test, final_prediction), end =" ")
        print( "precission",precision_score(y_test, final_prediction,average='macro'),"recall",recall_score(y_test, final_prediction,average='macro'), "f1 score",f1_score(y_test, final_prediction,average='macro'))
        #print(generated_counter , final_counter, end=" ")
        print("memory:",memory, end=" ")
        print("required_time:",time.time()-t1)


if __name__ == '__main__':
    start_time = time.time()
    test_uci()
    end_time=time.time()
    print("required_time:", end_time-start_time)