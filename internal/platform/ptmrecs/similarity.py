from .constraints import TimeConstraint
from .cache import TemporalCache
import pandas as pd


class TemporalPearson:

    def __init__(self, cache: TemporalCache, time_constraint: TimeConstraint = None, min_common_elements: int = 5):
        self.time_constraint = time_constraint
        self.cache = cache
        self.min_common_elements = min_common_elements

    def mean_centered_pearson(self, user_id, movie_id, k_neighbours: pd.DataFrame):
        from .trainset import TrainsetUser, TrainsetMovie
        trainset_user = TrainsetUser(cache=self.cache)
        trainset_movie = TrainsetMovie(cache=self.cache)

        # Calculate Mean Centered Prediction
        user_avg_rating = trainset_user.get_user_avg(user_id=user_id)

        weighted_sum = 0.0
        sum_of_weights = 0.0
        for neighbour_id, data in k_neighbours.iterrows():
            # Get each neighbour's correlation 'user_id' and her rating to 'movie_id'
            neighbour_corr = data['correlation']
            neighbour_rating = trainset_movie.get_movie_rating(movie_id=movie_id, user_id=neighbour_id)
            # If the neighbour doesnt give rating to the movie_id, pass this around of the loop
            if neighbour_rating == 0:
                continue
            neighbour_avg_rating = trainset_user.get_user_avg(user_id=neighbour_id)
            neighbour_mean_centered_rating = neighbour_rating - neighbour_avg_rating
            # Calculate Weighted sum and sum of weights
            weighted_sum += neighbour_mean_centered_rating * neighbour_corr
            sum_of_weights += neighbour_corr

        # Predict
        if sum_of_weights != 0:
            prediction_rating = user_avg_rating + (weighted_sum / sum_of_weights)
        else:
            prediction_rating = 0  # In this case, none of the neighbours have given rating to 'the movie'

        return prediction_rating

    def get_corr_matrix(self):

        if (self.cache.is_user_movie_matrix_cached and
                self.cache.does_cache_match(time_constraint=self.time_constraint,
                                            min_common_elements=self.min_common_elements)):
            if self.cache.is_temporal_cache_valid():
                return self.cache.user_correlations

        # Create and Cache the correlation matrix
        self.cache_corr_matrix()

        return self.cache.user_correlations

    def cache_corr_matrix(self):
        movie_ratings = self.cache.movie_ratings
        if self.time_constraint is None:
            user_movie_matrix = movie_ratings.pivot_table(index='title', columns='user_id', values='rating')
        elif self.time_constraint.is_valid_max_limit():
            user_movie_matrix = movie_ratings[
                (movie_ratings.timestamp < self.time_constraint.end_dt)].pivot_table(index='title',
                                                                                     columns='user_id',
                                                                                     values='rating')
        elif self.time_constraint.is_valid_time_bin():
            user_movie_matrix = movie_ratings[
                (movie_ratings.timestamp >= self.time_constraint.start_dt)
                & (movie_ratings.timestamp < self.time_constraint.end_dt)].pivot_table(index='title',
                                                                                       columns='user_id',
                                                                                       values='rating')
        else:
            raise Exception("Invalid time constraint!")

        # Cache the user_movie_matrix and temporal correlation matrix
        self.cache.user_movie_matrix = user_movie_matrix
        self.cache.is_user_movie_matrix_cached = True

        self.cache.user_correlations = user_movie_matrix.corr(method="pearson", min_periods=self.min_common_elements)
        self.cache.min_common_elements = self.min_common_elements
        self.cache.is_user_correlations_cached = True
