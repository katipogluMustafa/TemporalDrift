# -*- coding: utf-8 -*-
"""
Created on Thu May    3 10:22:34 2018
Updated on Tue March  4 10:51:25 2020

@Authors: Frank , Mustafa Katipoglu
@Style: Google Python Style
"""

from internal.platform.CoreEvaluator import RecommenderMetrics
from internal.platform.CoreEvaluator import EvaluationData

class EvaluatedAlgorithm:

    def __init__(self, algorithm, name):
        self.algorithm = algorithm
        self.name = name

    def evaluate(self, evaluation,data, do_top_n, n=10, verbose=True):
        sd = 5

    def get_name(self):
        return self.name

    def get_algorithm(self):
        return self.algorithm

