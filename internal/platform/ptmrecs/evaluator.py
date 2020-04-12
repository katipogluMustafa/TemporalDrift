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

    def evaluate_best_max_year_in_bulk(self, n,
                                       n_users, n_movies, k=10,
                                       min_year=-1,
                                       max_year=-1) -> dict:
        """
        Evaluate and collect data about best max year constraint which can be put instead of no constraint.

        This method calls 'evaluate_best_max_year_constraint' method 'n' times.
        Takes required precautions for bulk calling.

        :param n: Number of runs that we run the evaluate_best_max_year_constraint() method
        :param n_users: Number of users to check
        :param n_movies: Number of movies per user to check
        :param k: Number of neighbours of each user to take into account when making prediction
        :param min_year: First year to evaluate
        :param max_year: Last year to evaluate
        :return: (no_constrain_rmse_data, best_year_constraint_results)
        """
        if min_year == -1:
            min_year = self.trainset.trainset_user.get_first_timestamp().year

        if max_year == -1:
            max_year = datetime.now().year

        time_constraint = TimeConstraint(end_dt=datetime(year=min_year, month=1, day=1))
        # Create cache if bulk_corr_cache is allowed
        self.trainset.similarity.cache_user_corrs_in_bulk_for_max_limit(time_constraint,
                                                                        min_year=min_year,
                                                                        max_year=max_year)

        run_results = dict()
        for i in range(n):
            run_results[i] = self.evaluate_best_max_year_constraint(n_users=n_users, n_movies=n_movies, k=k,
                                                                    min_year=min_year, max_year=max_year,
                                                                    create_cache=False,)

        return run_results

    def evaluate_best_max_year_constraint(self, n_users, n_movies, k,
                                          max_diff=0.1,
                                          min_year=-1, max_year=-1,
                                          create_cache=True) -> defaultdict:
        """
        Evaluate the max_year constraint for evaluate_max_year_constraint method.

        :param max_diff: maximum difference between rmse when no constraint and with given year constraint.
        :param n_users: Number of users to evaluate
        :param n_movies: Number of movies per user to evaluate
        :param k: Number of neighbours of each user to take into account when making prediction
        :param min_year: First year to evaluate
        :param max_year: Last year to evaluate
        :param create_cache: create cache before running. For bulk callers.
        :return: Votes for years where each year got its vote
                 when rmse is less than 'max_diff' in between no constraint and year constraint
        """

        if min_year == -1:
            min_year = self.trainset.trainset_user.get_first_timestamp().year

        if max_year == -1:
            max_year = datetime.now().year

        if n_users > 600:
            user_list = self.trainset.trainset_user.get_users()  # No need to random selection, get all users
        else:
            user_list = self.trainset.trainset_user.get_random_users(n=n_users)  # Select random n users

        # Calculate RMSE With No Constraint
        no_constraint_data = dict()
        for user_id in user_list:
            rmse = Accuracy.rmse(self.trainset.predict_movies_watched(user_id, n_movies, k))
            no_constraint_data[user_id] = rmse

        # # Calculate RMSE With Time Constraint

        # Cache all years before processing
        time_constraint = TimeConstraint(end_dt=datetime(year=min_year, month=1, day=1))
        # Create cache if bulk_corr_cache is allowed
        if create_cache:
            self.trainset.similarity.cache_user_corrs_in_bulk_for_max_limit(time_constraint,
                                                                            min_year=min_year,
                                                                            max_year=max_year)
        # Votes to years is stored inside time_constraint_data
        time_constraint_data = defaultdict(int)
        for year in range(min_year, max_year):
            time_constraint.end_dt = time_constraint.end_dt.replace(year=year)

            for user_id in user_list:
                rmse = Accuracy.rmse(self.trainset.predict_movies_watched(user_id=user_id, n=n_movies, k=k,
                                                                          time_constraint=time_constraint))
                if abs(rmse - no_constraint_data[user_id]) < max_diff:
                    time_constraint_data[year] += 1

        return time_constraint_data

    def evaluate_max_year_constraint(self, n_users, n_movies, k, time_constraint):
        """
        Compare given time_constraint with normal where no constraint exists.

        Time constraint is of type max_year which means the system will be set to a certain year.

        :param n_users: Number of users to evaluate
        :param n_movies: Number of movies per user to evaluate
        :param k: Number of neighbours to take into account when making movie prediction
        :param time_constraint: Time constraint which will be applied.
        :return: DataFrame of results which contains rmse with constraint and no constraint, as well as runtime.
        """
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

    def evaluate_time_bins_in_bulk(self, n, n_users, k=10,
                                   min_year=-1,
                                   max_year=-1,
                                   min_time_bin_size=2, max_time_bin_size=10):
        """
        Evaluate time bins and return the results.

        This method calls 'evaluate_time_bins' method 'n' times. Takes required precautions for bulk calling.

        :param n: Number of runs
        :param n_users: Number of users
        :param k: Number of neighbours will be used when making prediction
        :param min_year: First year to start when taking time bins
        :param max_year: When to stop when taking time bins, last is not included.
        :param min_time_bin_size: Minimum bin size in years
        :param max_time_bin_size: Maximum bin size in years
        :return: Evaluation results
        """
        if min_year == -1:
            min_year = self.trainset.trainset_user.get_first_timestamp().year

        if max_year == -1:
            max_year = datetime.now().year

        # Cache all years before processing
        time_constraint = TimeConstraint(start_dt=datetime(year=min_year, month=1, day=1),
                                         end_dt=datetime(year=max_year, month=1, day=1))
        self.trainset.similarity.cache_user_corrs_in_bulk_for_time_bins(time_constraint,
                                                                        min_year=min_year,
                                                                        max_year=max_year,
                                                                        min_time_bin_size=min_time_bin_size,
                                                                        max_time_bin_size=max_time_bin_size)

        run_results = dict()
        for i in range(n):
            run_results[i] = self.evaluate_time_bins(n_users=n_users, k=k, min_year=min_year, max_year=max_year,
                                                     min_time_bin_size=min_time_bin_size,
                                                     max_time_bin_size=max_time_bin_size,
                                                     create_cache=False)

        return run_results

    def evaluate_time_bins(self, n_users, k, min_year=-1, max_year=-1,
                           min_time_bin_size=2, max_time_bin_size=10,
                           create_cache=True) -> dict:
        """

        :param n_users: Number of users
        :param k: Number of neighbours will be used when making prediction
        :param min_year: First year to start when taking time bins
        :param max_year: When to stop when taking time bins, last is not included.
        :param min_time_bin_size: Minimum bin size in years
        :param max_time_bin_size: Maximum bin size in years
        :param create_cache: Create cache before calling time bins. For bulk callers.
        :return:
        """
        trainset = self.trainset

        if min_year == -1:
            min_year = self.trainset.trainset_user.get_first_timestamp().year

        if max_year == -1:
            max_year = datetime.now().year

        if n_users > 600:
            user_list = trainset.trainset_user.get_users()
        else:
            user_list = trainset.trainset_user.get_random_users(n=n_users)
        user_movie_list = trainset.trainset_movie.get_random_movie_per_user(user_list)
        data = dict()

        result = list()

        if create_cache:
            # Cache all years before processing
            time_constraint = TimeConstraint(start_dt=datetime(year=min_year, month=1, day=1),
                                             end_dt=datetime(year=max_year, month=1, day=1))
            self.trainset.similarity.cache_user_corrs_in_bulk_for_time_bins(time_constraint,
                                                                            min_year=min_year,
                                                                            max_year=max_year,
                                                                            min_time_bin_size=min_time_bin_size,
                                                                            max_time_bin_size=max_time_bin_size)

        # Take each bins where first bin 'min_time_bin_size' years, last one 'max_time_bin_size - 1' years
        for time_bin_size in range(min_time_bin_size, max_time_bin_size):
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
                                                                                  end_dt=datetime(curr_year+time_bin_size, 1, 1)),
                                                   bin_size=time_bin_size)
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
                                     "runtime": runtime
                                     }
                result.append(iteration_results)

        data['result'] = result
        return data