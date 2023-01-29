#!/usr/bin/env python3

import math


class Rule:
    """ Represents a final Rule that is extracted from the tree. """

    def __init__(self, items, label, confidence, ss, support):
        self._items = items
        self._label = label
        self._confidence = confidence
        self._ss = ss
        self._support = support

    def get_items(self):
        return self._items

    def get_label(self):
        return self._label

    def get_confidence(self):
        return self._confidence

    def get_ss(self):
        return self._ss

    def get_support(self):
        return self._support

        
    def set_items(self, new_items):
        self._items = new_items
 
    def str_print(self,temp):
        try:
            x=[]
            for i in self._items:
                x.append(temp[i])
            return "{} {};{:.4f},{:.3f},{:.3f}".format(' '.join(map(str, x)),
                                          self._label,
                                          self._support,
                                          self._confidence,
                                          math.log(self._ss),
                                          )
        except Exception as e:
            print(repr(e), self._items, self._label, self._ss)
            return "{} {};{:.4f},{:.3f},{:.3f}".format(' '.join(map(str, x)),
                                          self._label,
                                          float(self._support),
                                          float(self._confidence),
                                          0.0,
                                          )


    def __str__(self):
        try:
          
            return "{} -> {};({:.4f},{:.3f},{:.3f})".format(' '.join(map(str, self._items)),
                                          self._label,
                                          self._support,
                                          self._confidence,
                                          math.log(self._ss),
                                          )
        except Exception as e:
            print(repr(e), self._items, self._label, self._ss)
            return "{} {};{:.4f},{:.3f},{:.3f}".format(' '.join(map(str, self._items)),
                                          self._label,
                                          float(self._support),
                                          float(self._confidence),
                                          0.0,
                                          )


