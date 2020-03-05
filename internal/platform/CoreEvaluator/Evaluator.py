# -*- coding: utf-8 -*-
"""
Created on Thu May    3 10:22:34 2018
Updated on Tue March  4 12:14:25 2020

@Authors: Frank , Mustafa Katipoglu
@Style: Google Python Style
"""

from internal.platform.CoreEvaluator import EvaluationData
from internal.platform.CoreEvaluator import EvaluatedAlgorithm

class Evaluator:

    algorithms = []

    def __init__(self, dataset, rankings):
        evaluation_data = EvaluationData(dataset, rankings)
        self.dataset = evaluation_data

    def add_algorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)

    def evaluate(self, do_top_n):
        results = {}
        for algorithm in self.algorithms:
            print("Evaluating ", algorithm.get_name(), "...")
            results[algorithm.get_name()] = algorithm.Evaluate(self.dataset, do_top_n)

        Evaluator.print_evaluation_results(results, do_top_n)

    @staticmethod
    def print_evaluation_results(results, do_top_n):
        if do_top_n:
            print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
                "Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage", "Diversity", "Novelty"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}".format(
                    name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"],
                    metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]))
        else:
            print("{:<10} {:<10} {:<10}".format("Algorithm", "RMSE", "MAE"))
            for (name, metrics) in results.items():
                print("{:<10} {:<10.4f} {:<10.4f}".format(name, metrics["RMSE"], metrics["MAE"]))

        print("\nLegend:\n")
        print("RMSE:      Root Mean Squared Error. Lower values mean better accuracy.")
        print("MAE:       Mean Absolute Error. Lower values mean better accuracy.")
        if do_top_n:
            print("HR:        Hit Rate; how often we are able to recommend a left-out rating. Higher is better.")
            print(
                "cHR:       Cumulative Hit Rate; hit rate, confined to ratings above a certain threshold. Higher is better.")
            print(
                "ARHR:      Average Reciprocal Hit Rank - Hit rate that takes the ranking into account. Higher is better.")
            print(
                "Coverage:  Ratio of users for whom recommendations above a certain threshold exist. Higher is better.")
            print(
                "Diversity: 1-S, where S is the average similarity score between every possible pair of recommendations")
            print("           for a given user. Higher means more diverse.")
            print("Novelty:   Average popularity rank of recommended items. Higher means more novel.")


    def sample_top_n_recs(self, ml, test_subject=85, k=10):     # TODO: Review This k param
        for algo in self.algorithms:
            print("\nUsing recommender ", algo.get_name())

            print("\nBuilding recommendation model...")
            train_set = self.dataset.get_full_train_set()
            algo.get_algorithm().fit(train_set)

            print("Computing recommendations...")
            test_set = self.dataset.get_anti_test_set_for_user(test_subject)
            predictions = algo.get_algorithm.test(test_set)
            recommendations = []

            print("\nWe recommend:")
            for user_id, movie_id, actual_rating, estimated_rating, _ in predictions:
                int_movie_id = int(movie_id)
                recommendations.append((int_movie_id, estimated_rating))

            recommendations.sort(key=lambda x: x[1], reverse=True)

            for ratings in recommendations[:10]:
                print(ml.get_movie_name(ratings[0]), ratings[1])