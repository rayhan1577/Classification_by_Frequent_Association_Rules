#!/usr/bin/env python3

import random

import numpy as np

import config

random.seed(config.random_seed)
np.random.seed(config.random_seed)


class _TreeLayer:
    """Holds all data for a layer of the tree:
        a set of numpy arrays holding data for each node in the layer: _data*,
            they are created when they are needed (add_initial* vs add_final* methods)
            and are deleted when they are not needed.
        a set of numpy arrays holding pointers for each layer to the next layer _pointers*
            _pointers is a matrix holding a row for each node in the tree, and each column
            corresponds to an item (aka feature)
            __pointers_dead is used to show if we should stop traversing from that node
        if _has_base is set to true (ie, many many nodes in the layer, threshold in config.py)
            the index in the next layer is _pointers[x,y] + _pointers_base[x] to use 
            np.uint8 in the former matrix.
    """

    def __init__(self, labels_count, best_int, features_count):

        if features_count < 2**8-1:
            self._small_int = np.uint8
        elif features_count < 2**16-1:
            self._small_int = np.uint16
        else:
            raise Exception("There shouldn't be more than 64k features in the dataset.")

        self._labels_count = labels_count
        self._best_int  = best_int
        self._big_int   = np.int32

        self._best_float = np.float64
        self._features_count = features_count
        
        self._has_base = False
        self.__counter = np.int32(0)

    def get_counter(self):
        """ return the counter (ie, how many nodes in this layer) """
        return self.__counter

    def set_has_base(self):
        """ this layer used base_value for pointers"""
        self._has_base = True

    def has_base(self):
        """ check if this layer used base_value for pointers"""
        return self._has_base

    def add_pointers_capacity(self):

        size = self.__counter

        if self._has_base:
            self._pointers_base = np.zeros((size), dtype=self._big_int)
            self._pointers      = np.zeros((size, self._features_count), dtype=self._small_int)
        else:
            self._pointers      = np.zeros((size, self._features_count), dtype=self._big_int)


    def add_initial_capacity(self, size):
        
        assert size >= 0
        
        self._data_antecedent = np.zeros((size), dtype=self._best_int)
        self._data_subsequent = np.zeros((size, self._labels_count), dtype=self._best_int)

    def add_final_capacity(self, size):
        
        assert size >= 0
        
        self._data_ispss     = np.zeros((size, self._labels_count), dtype=np.bool_)
        self._data_ss        = np.zeros((size, self._labels_count), dtype=self._best_float)
        self._data_minss     = np.ones((size, self._labels_count), dtype=self._best_float)
        self._data_isminimal = np.zeros((size, self._labels_count), dtype=np.bool_)

        self._pointers_dead  = np.zeros((size), dtype=np.bool_)

    def release_initial_capacity(self):

        del self._data_antecedent
        del self._data_subsequent
        del self._data_ss
        del self._data_isminimal

    def release_further_capacity(self):
        
        del self._data_minss
        del self._data_ispss

    def add_node(self, label_index):
        """ returns the index to this node, used for first layer"""

        # there is no arrays for this layer
        if self.__counter == 0:
            self.add_initial_capacity(1)
            self.add_final_capacity(1)

        # we need to add further capacity to this layer
        if self.__counter >= self._data_antecedent.shape[0]:
            self._update_capacity(1)

        # increase counter of this node
        self._data_antecedent[self.__counter] = 1

        # increase counter of this rule_node
        self._data_subsequent[self.__counter, label_index] = 1

        self.__counter += 1

        return self.__counter-1

    def _update_capacity(self, size):
        # only first layer
        
        assert size >= 0
        
        self._data_antecedent.resize((self._data_antecedent.shape[0]+size), refcheck=False)
        self._data_subsequent.resize((self._data_subsequent.shape[0]+size, self._data_subsequent.shape[1]), refcheck=False)
        self._data_ispss.resize((self._data_ispss.shape[0]+size, self._data_ispss.shape[1]), refcheck=False)
        self._data_ss.resize((self._data_ss.shape[0]+size, self._data_ss.shape[1]), refcheck=False)
        self._data_minss.resize((self._data_minss   .shape[0]+size, self._data_minss.shape[1]), refcheck=False)
        self._data_isminimal.resize((self._data_isminimal.shape[0]+size, self._data_isminimal.shape[1]), refcheck=False)

        self._pointers_dead.resize((self._pointers_dead.shape[0]+size), refcheck=False)

    def update_node(self, node_index, label_index):
        """update an existing node, used for first layer"""

        # increase counter of this node
        self._data_antecedent[node_index] += 1

        # increase counter of this rule_node
        self._data_subsequent[node_index, label_index] += 1

    def update_node_np(self, label_count_arr):
        """ add a new node to the tree by 
            updating the antecedent and subsequent values for this node.
        """

        # set the antecedent count of this node
        self._data_antecedent[self.__counter] = label_count_arr.sum()

        # set the subsequent count for all labels
        self._data_subsequent[self.__counter] = label_count_arr
    
        self.__counter += 1

        return self.__counter-1
