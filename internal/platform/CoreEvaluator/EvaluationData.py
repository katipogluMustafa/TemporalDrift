# -*- coding: utf-8 -*-
"""
Created on Thu May    3 10:22:34 2018
Updated on Tue March  5 06:43:17 2020

@Authors: Frank , Mustafa Katipoglu
@Style: Google Python Style
"""

from surprise.model_selection import train_test_split
from surprise.model_selection import LeaveOneOut
from surprise import KNNBaseline

class EvaluationData:

    def __init__(self, data, popularity_rankings):

        self.popularity_rankings = popularity_rankings

        # Build a full training set for evaluating overall properties
        self.full_train_set = data.build_full_trainset()
        self.full_anti_test_set = self.full_train_set.build_anti_testset()

        # Build a 75/25 train/test split for measuring accuracy
        self.train_set, self.test_set = train_test_split(data, test_size=.25, random_state=1)

        # Build a "leave one out" train/test split for evaluating top-N recommender
        # And build an anti-test-set for building predictions
        loocv = LeaveOneOut(n_splits=1, random_state=1)
        for train, test in loocv.split(data):
            self.loocv_train = train
            self.loocv_test = test

        self.loocv_anti_test_set = self.loocv_train.build_anti_testset()

        # Compute similarity matrix between items so we can measure diversity
        sim_options = {'name': 'cosine', 'user_based':False}
        self.sims_algo = KNNBaseline(sim_options=sim_options)
        self.sims_algo.fit(self.full_train_set)

    def get_full_train_set(self):
        return self.full_train_set

    def get_full_anti_test_set(self):
        return self.full_anti_test_set

    def get_anti_test_set_for_user(self, test_subject):
        train_set = self.full_train_set
        fill = train_set.global_mean
        anti_test_set = []
        user = train_set.to_inner_uid(str(test_subject))
        user_items = set([j for (j, _) in train_set.ur[user]])
        anti_test_set += [
                         (train_set.to_raw_uid(user), train_set.to_raw_iid(i), fill)
                          for i in train_set.all_items()
                          if i not in user_items
                         ]
        return anti_test_set

    def get_train_set(self):
        return self.train_set

    def get_test_set(self):
        return self.test_set

    def get_loocv_train_set(self):
        return self.loocv_train

    def get_loocv_test_set(self):
        return self.loocv_test

    def get_loocv_anti_test_set(self):
        return self.loocv_anti_test_set

    def get_similarities(self):
        return self.sims_algo

    def get_popularity_rankings(self):
        return self.popularity_rankings

