# -*- coding: utf-8 -*-
"""
Created on Thu May    3 10:22:34 2018
Updated on Tue March  4 12:14:25 2020

@Authors: Frank , Mustafa Katipoglu
@Style: Google Python Style
"""

from internal.platform.CoreEvaluator.MovieLens import MovieLens
from internal.platform.CoreEvaluator.Evaluator import Evaluator
from surprise import SVD
from surprise import NormalPredictor
import numpy as np
import random

def load_movielens_data():
    ml = MovieLens()

    print("Loading movie ratings...")
    data = ml.load_movielens_latest_small()

    print("\nComputing movie popularity ranks so wen measure novelty...")
    rankings = ml.get_popularity_ranks()

    return data, rankings

np.random.seed(0)
random.seed(0)

evaluation_data, rankings = load_movielens_data()

evaluator = Evaluator(evaluation_data, rankings)

SVDAlgorithm = SVD(random_state=10)
evaluator.add_algorithm(SVDAlgorithm, "SVD")

normalAlg = NormalPredictor()
evaluator.add_algorithm(normalAlg, "Random")

evaluator.evaluate(True)