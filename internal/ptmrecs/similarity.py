from .constraints import TimeConstraint
import pandas as pd
from .cache import TemporalCache
from datetime import datetime


class TemporalPearson:

    def __init__(self, cache: TemporalCache, time_constraint: TimeConstraint = None, min_common_elements: int = 5):
        self.time_constraint = time_constraint
        self.cache = cache
        self.min_common_elements = min_common_elements
        from .trainset import TrainsetUser, TrainsetMovie
        self.trainset_user = TrainsetUser(cache=self.cache)
        self.trainset_movie = TrainsetMovie(cache=self.cache)

    def mean_centered_pearson(self, user_id, movie_id, k_neighbours: pd.DataFrame) -> float:
        """
        Make Mean Centered Prediction

        :param user_id: user of interest
        :param movie_id: this movie's rating is the one we we want to predict
        :param k_neighbours: k nearest neighbours in DataFrame where its index user_id, column correlation in between.
        :return: Prediction rating
        """
        # If a movie with movie_id not exists, predict 0
        if self.trainset_movie.get_movie(movie_id=movie_id).empty:
            return 0

        if k_neighbours is None or k_neighbours.empty:
            return 0

        user_avg_rating = self.trainset_user.get_user_avg(user_id=user_id)

        weighted_sum = 0.0
        sum_of_weights = 0.0
        for neighbour_id, data in k_neighbours.iterrows():
            
            # Get each neighbour's correlation 'user_id' and her rating to 'movie_id'
            neighbour_corr = data['correlation']
            neighbour_rating = self.trainset_movie.get_movie_rating(movie_id=movie_id, user_id=neighbour_id)
            
            # If the neighbour has not given rating to this movie, pass this round of the loop
            if neighbour_rating == 0:
                continue
            neighbour_avg_rating = self.trainset_user.get_user_avg(user_id=neighbour_id)
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

    def get_corr_matrix(self, bin_size=-1):
        user_corrs = None
        
        # if valid cache found, try to get user corrs from there
        if self.cache.is_temporal_cache_valid():
            
            # First check user-correlations cache
            user_corrs = self.cache.get_user_corrs(self.min_common_elements, self.time_constraint)
            if user_corrs is not None:
                return user_corrs
            
            # Then check bulk-user-correlations
            user_corrs = self.cache.get_user_corrs_from_bulk(time_constraint=self.time_constraint,
                                                             min_common_elements=self.min_common_elements,
                                                             bin_size=bin_size)
            if user_corrs is not None:
                return user_corrs

        # we are here, if cache not found or no cache match

        # Create user correlations
        user_corrs = TemporalPearson.create_user_corrs(movie_ratings=self.cache.movie_ratings,
                                                       time_constraint=self.time_constraint,
                                                       min_common_elements=self.min_common_elements)
        
        # Cache the user_corrs, this will only work when caching is activated.
        self.cache.set_user_corrs(user_corrs=user_corrs,
                                  min_common_elements=self.min_common_elements,
                                  time_constraint=self.time_constraint)

        return user_corrs

    @staticmethod
    def create_user_corrs(movie_ratings, time_constraint: TimeConstraint, min_common_elements):
        # by default movie_ratings is for no time constraint
        # with these controls change the time constraint of the movie_ratings
        if time_constraint is not None:
            if time_constraint.is_valid_max_limit():
                movie_ratings = movie_ratings[movie_ratings.timestamp < time_constraint.end_dt]
            elif time_constraint.is_valid_time_bin():
                movie_ratings = movie_ratings[(movie_ratings.timestamp >= time_constraint.start_dt)
                                              & (movie_ratings.timestamp < time_constraint.end_dt)]

        user_movie_matrix = movie_ratings.pivot_table(index='title', columns='user_id', values='rating')
        return user_movie_matrix.corr(method="pearson", min_periods=min_common_elements)

    def cache_user_corrs_in_bulk_for_max_limit(self, time_constraint: TimeConstraint, min_year, max_year):
        """
        Cache user correlations by changing year of the time_constraint
        for each year in between min_year and max_year(not included)

        :param time_constraint: time_constraint apply
        :param min_year: start of the range
        :param max_year: end of the range
        """

        if self.cache.use_bulk_corr_cache:
            if time_constraint is not None and time_constraint.is_valid_max_limit():
                self.cache.user_corrs_in_bulk = dict()
                for year in range(min_year, max_year):
                    time_constraint.end_dt = time_constraint.end_dt.replace(year=year)
                    corrs = TemporalPearson.create_user_corrs(self.cache.movie_ratings, time_constraint,
                                                              self.min_common_elements)
                    self.cache.user_corrs_in_bulk[year] = corrs
            else:
                raise Exception("Trying to cache user correlations in bulk for max_limit "
                                "but start time is not max_limit!")
        else:
            raise Exception("Trying to create bulk corr cache when use_bulk_corr_cache is False")

    def cache_user_corrs_in_bulk_for_time_bins(self, time_constraint: TimeConstraint, min_year, max_year,
                                               min_time_bin_size=2, max_time_bin_size=10):
        if self.cache.use_bulk_corr_cache:
            if time_constraint is not None and time_constraint.is_valid_time_bin():
                del self.cache.user_corrs_in_bulk    # invalidate old cache
                self.cache.user_corrs_in_bulk = dict()
                for time_bin_size in range(min_time_bin_size, max_time_bin_size):
                    self.cache.user_corrs_in_bulk[time_bin_size] = dict()
                    for shift in range(0, time_bin_size):
                        curr_year = min_year + shift
                        while (curr_year + time_bin_size) < max_year:
                            time_constraint = TimeConstraint(start_dt=datetime(curr_year, 1, 1),
                                                             end_dt=datetime(curr_year + time_bin_size, 1, 1))
                            corrs = TemporalPearson.create_user_corrs(self.cache.movie_ratings,
                                                                      time_constraint,
                                                                      self.min_common_elements)
                            self.cache.user_corrs_in_bulk[time_bin_size][curr_year] = corrs
                            curr_year += time_bin_size
        else:
            raise Exception("Trying to create bulk corr cache when use_bulk_corr_cache is False")

    @property
    def time_constraint(self):
        return self._time_constraint

    @time_constraint.setter
    def time_constraint(self, value):
        self._time_constraint = value

