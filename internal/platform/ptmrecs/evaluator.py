from .trainset import Trainset
from datetime import datetime
from collections import defaultdict
from .accuracy import Accuracy
from .constraints import TimeConstraint
from timeit import default_timer
import random
import pandas as pd


class Evaluator:

    def __init__(self, trainset: Trainset):
        self.trainset = trainset

    def evaluate_best_max_year_constraint(self, n_users, n_movies, k, rmse_diff):
        """
        Evaluate the TimeConstraints and No TimeConstraints and get data about best max year constraint.

        :param n_users: Number of users to evaluate
        :param n_movies: Number of movies to evaluate per user
        :param k: number of neighbours to take into account
        :param rmse_diff: Max RMSE difference to record the result
        :return: Dictionary of evaluation results
        """
        trainset = self.trainset
        min_year = trainset.trainset_user.get_first_timestamp().year
        max_year = datetime.now().year

        if n_users > 600:
            user_list = trainset.trainset_user.get_users()  # No need to random selection, get all users
        else:
            user_list = trainset.trainset_user.get_random_users(n=n_users)  # Select random n users

        data = defaultdict(list)

        # For Each Year
        for year in range(min_year, max_year):
            for user_id in user_list:
                # Calculate RMSE
                rmse = Accuracy.rmse(
                    trainset.predict_movies_watched(user_id=user_id, n=n_movies, k=k, time_constraint=None))
                time_constrained_rmse = Accuracy.rmse(
                    trainset.predict_movies_watched(user_id=user_id, n=n_movies, k=k,
                                                    time_constraint=TimeConstraint(end_dt=datetime(year=year,
                                                                                                   month=1,
                                                                                                   day=1))))
                if rmse != 0 and time_constrained_rmse != 0 and abs(rmse - time_constrained_rmse) <= rmse_diff:
                    data[year].append((rmse, time_constrained_rmse))

        return data

    def evaluate_max_year_constraint(self, n_users, n_movies, k, time_constraint):
        trainset = self.trainset
        data = list()

        for i in range(n_users):
            # Get Random User
            user_id = random.randint(1, 610)
            # Predict movies for user and record runtime
            st = default_timer()
            rmse = Accuracy.rmse(
                trainset.predict_movies_watched(user_id=user_id, n=n_movies, k=k, time_constraint=None))
            r1 = default_timer() - st
            # Predict movies with time_constraint for user and record runtime
            st = default_timer()
            time_constrained_rmse = Accuracy.rmse(
                trainset.predict_movies_watched(user_id=user_id, n=n_movies, k=k, time_constraint=time_constraint))
            r2 = default_timer() - st
            # Save iteration data
            data.append([user_id, rmse, r1, time_constrained_rmse, r2])

        data = pd.DataFrame(data)
        data.columns = ['user_id', 'rmse', 'runtime1', 'temporal_rmse', 'runtime2']
        data.set_index('user_id', inplace=True)
        return data

    def evaluate_time_bins(self, n_users, k):
        trainset = self.trainset
        min_year = trainset.trainset_user.get_first_timestamp().year
        max_year = datetime.now().year

        if n_users > 600:
            user_list = trainset.trainset_user.get_users()
        else:
            user_list = trainset.trainset_user.get_random_users(n=n_users)
        user_movie_list = trainset.trainset_movie.get_random_movie_per_user(user_list)
        data = {"n_users": n_users, "k_neighbours": k}

        result = list()

        # Take each bins where first bin 2 years, last one 9 years
        for time_bin_size in range(2, 10):
            # Shift each time_bin starting with 0 years up until (time_bin-1) years
            for shift in range(0, time_bin_size):
                curr_year = min_year + shift
                predictions = list()
                start_time = default_timer()
                # Scan and make predictions for all the time_bins
                while (curr_year + time_bin_size) < max_year:
                    for user_id, movie_id in user_movie_list:
                        p = trainset.predict_movie(user_id=user_id, movie_id=movie_id, k=k,
                                                   time_constraint=TimeConstraint(start_dt=datetime(curr_year, 1, 1),
                                                                                  end_dt=datetime(curr_year+time_bin_size, 1, 1)))
                        # if prediction has been done successfully
                        if p != 0:
                            r = trainset.trainset_movie.get_movie_rating(movie_id=movie_id, user_id=user_id)
                            # Append (prediction, actual_rating)
                            predictions.append((p, r))
                    curr_year += time_bin_size
                runtime = default_timer() - start_time
                bin_rmse = Accuracy.rmse(predictions=predictions)
                iteration_results = {"bin_size": time_bin_size,
                                     "start_year": min_year + shift,
                                     "predictions": predictions,
                                     "rmse": bin_rmse,
                                     "runtime": runtime}
                result.append(iteration_results)

        data['result'] = result
        return data