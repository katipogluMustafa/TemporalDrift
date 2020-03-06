# -*- coding: utf-8 -*-
"""
Created on Thu May    3 10:22:34 2018
Updated on Tue March  4 10:51:25 2020

@Authors: Frank , Mustafa Katipoglu
@Style: Google Python Style
"""

import itertools

from surprise import accuracy
from collections import defaultdict

class RecommenderMetrics: #TODO: Update Docs
    """
        Utility Class to provide recommendation metrics' calculation methods
    """

    @staticmethod
    def mean_absolute_error(predictions):
        """
        Wrapper method for calculating mae without printing to the console.
        :param: predictions: (list of Prediction) A list of predictions, as returned by the test() method.
        :return: Mean Absolute Error of predictions
        :raises: ValueError: When predictions is empty.
        """
        return accuracy.mae(predictions, verbose=False)         # verbose – If True, will print computed value. Default is True.

    @staticmethod
    def root_mean_square_error(predictions):
        """
        Wrapper method for calculating rmse without printing to the console.
        :param predictions: (list of Prediction) A list of predictions, as returned by the test() method.
        :return: Root Mean Square Error of predictions
        :raises: ValueError: When predictions is empty.
        """
        return accuracy.rmse(predictions, verbose=False)  # verbose – If True, will print computed value. Default is True.

    @staticmethod
    def get_top_n(predictions, n=10, minimum_rating=4.0):
        """
        Get Top n of the predictions after pre-filtering with minimum_rating
        :param predictions: (list of Prediction) A list of predictions, as returned by the test() method.
        :param n: Highest n number of predictions will be returned.
        :param minimum_rating: For pre-filtering predictions.
               Ignores those that doesnt have more than minimum_rating in top_n calculation.
        :return: top_n which is a defaultdict of user_id->[higher n piece predicted ratings]
        """
        top_n = defaultdict(list)

        # Filter those that only have greater predicted_rating than minimum_rating
        for user_id, movie_id, actual_rating, predicted_rating, _ in predictions:
            if predicted_rating >= minimum_rating:
                top_n[int(user_id)].append((int(movie_id),predicted_rating))

        # For each user, Sort predicted_rating in descending order and get highest n predictions
        for user_id, ratings in top_n.items():
            ratings.sort(key=lambda x: x[1], reverse=True)
            top_n[int(user_id)] = ratings[:n]

        return top_n


    @staticmethod
    def hit_rate(top_n_predicted, left_out_predictions) -> float:
        """
        Calculate Hit Rate
        :param top_n_predicted:
        :param left_out_predictions:
        :return: Hit Rate
        """
        hits = 0
        total = 0

        for left_out in left_out_predictions:
            user_id = left_out[0]
            left_out_movie_id = left_out[1]
            hit = False
            for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                if int(left_out_movie_id) == int(movie_id):
                    hit = True
                    break
            if hit :
                hit += 1

            total += 1

        return hits / total


    @staticmethod
    def cumulative_hit_rate(top_n_predicted, left_out_predictions, rating_cutoff=0) -> float:
        """
        Calculate Cumulative Hit Rate
        :param top_n_predicted:
        :param left_out_predictions:
        :param rating_cutoff:
        :return:
        """
        hits = 0
        total = 0

        for user_id, left_out_movie_id, actual_rating, _, _ in left_out_predictions:
            if actual_rating >= rating_cutoff:
                hit = False
                for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                    if left_out_movie_id == movie_id:
                        hit = True
                        break
                if hit :
                    hits += 1

                total += 1

        return hits/total


    @staticmethod
    def rating_hit_rate(top_n_predicted, left_out_predictions) -> None:
        """

        :param top_n_predicted:
        :param left_out_predictions:
        """
        hits = defaultdict(float)
        total = defaultdict(float)

        for user_id, left_out_movie_id, actual_rating, _, _ in left_out_predictions:
            hit = False
            for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                if int(left_out_movie_id) == movie_id:
                    hit = True
                    break
            if hit :
                hits[actual_rating] += 1

            total[actual_rating] += 1

        for rating in sorted(hits.keys()):
            print(rating, hits[rating] / total[rating])


    @staticmethod
    def novelty(top_n_predicted, rankings) -> float:
        n = 0
        total = 0
        for user_id in top_n_predicted.keys():
            for rating in top_n_predicted[user_id]:
                movie_id = rating[0]
                rank = rankings[movie_id]
                total += rank
                n += 1
        return total / n

    @staticmethod
    def diversity(top_n_predicted, similarity_algo) -> float:
        n = 0
        total = 0
        similarity_matrix = similarity_algo.compute_similarities()
        for user_id in top_n_predicted.keys():
            pairs = itertools.combinations(top_n_predicted[user_id], 2)
            for pair in pairs:
                movie_1 = pair[0][0]
                movie_2 = pair[1][0]
                inner_id_1 = similarity_algo.trainset.to_inner_iid(str(movie_1))
                inner_id_2 = similarity_algo.trainset.to_inner_iid(str(movie_2))
                similarity = similarity_matrix[inner_id_1][inner_id_2]
                total += similarity
                n += 1

        s = total / n
        return 1 - s

    @staticmethod
    def user_coverage(top_n_predicted, num_of_users, rating_threshold=0) -> float:
        hits = 0
        for user_id in top_n_predicted.keys():
            hit = False
            for movie_id, predicted_rating in top_n_predicted[user_id]:
                if predicted_rating >= rating_threshold:
                    hit = True
                    break
            if hit:
                hits += 1

        return hits / num_of_users


    @staticmethod
    def average_reciprocal_hit_rank(top_n_predicted, left_out_predictions) -> float:
        summation = 0
        total = 0

        for user_id, left_out_movie_id, actual_rating, estimated_rating,_ in left_out_predictions:
            hit_rank = 0
            rank = 0

            for movie_id, predicted_rating in top_n_predicted[int(user_id)]:
                rank += 1
                if int(left_out_movie_id) == movie_id:
                    hit_rank = rank
                    break

            if hit_rank > 0:
                summation += 1.0 / hit_rank

            total += 1

        return summation / total