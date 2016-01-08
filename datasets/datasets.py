__author__ = 'edhenry'

"""
Base IO code for toy datasets

Some features and functionality borrowed from scikit-learn implementation
"""

import csv
from os.path import dirname
from os.path import join


import numpy as np


class Bunch(dict):
    def __init__(self, **kwargs):
        super(Bunch, self).__init__(kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __getstate__(self):
        return self.__dict__

    def load_ctu(self, filename):
        """
        Load and return the CTU-13 netflow captures housed in CTU-13 directory


        ==================   ===============
        Classes                          113
        Samples per class            Unknown
        Samples total        Class dependent
        Dimensionality                    15
        Features              real, positive
        ==================   ===============

        :return:

        data : Dict
            Dictionary object, the interesting attributes are:
            'data', the data to learn, 'target' the classification labels,
            'target_names', meaning of labels, 'feature_names', the
            meaning of the features, and 'DESC', the full description of
            the dataset

        Examples:

        #TODO fill in examples with data

        """

        module_path = dirname(__file__)
        with open(join(module_path, 'CTU-13', filename)) as csv_file:
            data_file = csv.reader(csv_file)
            temp = next(data_file)
            features = temp[0:15]

            flowdata = list(data_file)

            flow_list = []

            for flow in flowdata:
                flow_list.append(tuple(flow))

        return Bunch(data = flow_list, features = features)