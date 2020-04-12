from .constraints import TimeConstraint


class Cache:
    """
    Cache the repetitive data when making predictions.
    """

    def __init__(self,
                 is_ratings_cached=False,
                 ratings=None,
                 is_movies_cached=False,
                 movies=None,
                 is_movie_ratings_cached=False,
                 movie_ratings=None,
                 is_user_movie_matrix_cached=False,
                 user_movie_matrix=None,
                 is_user_correlations_cached=False,
                 user_correlations=None,
                 min_common_elements=5,
                 use_avg_ratings_cache=True):
        """ Cached data is only valid when the boolean specifier is True """

        # 30% performance
        self.is_ratings_cached = is_ratings_cached
        self.ratings = ratings

        # 7 fold performance gain on 'movie' related queries
        self.is_movies_cached = is_movies_cached
        self.movies = movies

        self.is_movie_ratings_cached = is_movie_ratings_cached
        self.movie_ratings = movie_ratings

        self.is_user_movie_matrix_cached = is_user_movie_matrix_cached
        self.user_movie_matrix = user_movie_matrix

        self.is_user_correlations_cached = is_user_correlations_cached
        self.user_correlations = user_correlations

        self.min_common_elements = min_common_elements

        # if use avg ratings cache, on average 10 fold performance gain
        self.use_avg_ratings_cache = use_avg_ratings_cache
        if self.use_avg_ratings_cache:
            self.avg_user_ratings = self.create_user_avg_rating_cache()
        else:
            self.avg_user_ratings = None

    def create_user_avg_rating_cache(self):
        """
        Create cache of average ratings of each user
        :return: DataFrame of average ratings
        """
        if self.is_ratings_cached:
            data = self.ratings
        else:
            data = self.movie_ratings
        return data.groupby('user_id')[['rating']].mean()

    def get_user_corrs(self, min_common_elements, time_constraint=None):
        """
        If user correlations are cached returns the cache, else none
        :param min_common_elements: min common element in between users in order them to become neighbours
        :param time_constraint: used in temporal caches only, None in this context
        :return: user correlation matrix if cache found, else None
        """
        if self.is_user_correlations_cached:
            if self.min_common_elements == min_common_elements:
                return self.user_correlations
        return None

    # Properties
    @property
    def ratings(self):
        return self._ratings

    @ratings.setter
    def ratings(self, value):
        self._ratings = value

    @property
    def movies(self):
        return self._movies

    @movies.setter
    def movies(self, value):
        self._movies = value

    @property
    def movie_ratings(self):
        return self._movie_ratings

    @movie_ratings.setter
    def movie_ratings(self, value):
        self._movie_ratings = value

    @property
    def user_movie_matrix(self):
        return self._user_movie_matrix

    @user_movie_matrix.setter
    def user_movie_matrix(self, value):
        self._user_movie_matrix = value

    @property
    def user_correlations(self):
        return self._user_correlations

    @user_correlations.setter
    def user_correlations(self, value):
        self._user_correlations = value

    @property
    def min_common_elements(self):
        return self._min_common_elements

    @min_common_elements.setter
    def min_common_elements(self, value):
        self._min_common_elements = value


class TemporalCache(Cache):

    def __init__(self,
                 time_constraint: TimeConstraint,
                 is_ratings_cached=False,
                 ratings=None,
                 is_movies_cached=False,
                 movies=None,
                 is_movie_ratings_cached=False,
                 movie_ratings=None,
                 is_user_movie_matrix_cached=False,
                 user_movie_matrix=None,
                 is_user_correlations_cached=False,
                 user_correlations=None,
                 min_common_elements=5,
                 use_avg_ratings_cache=True,
                 use_bulk_corr_cache=True):

        super().__init__(is_ratings_cached=is_ratings_cached,
                         ratings=ratings,
                         is_movies_cached=is_movies_cached,
                         movies=movies,
                         is_movie_ratings_cached=is_movie_ratings_cached,
                         movie_ratings=movie_ratings,
                         is_user_movie_matrix_cached=is_user_movie_matrix_cached,
                         user_movie_matrix=user_movie_matrix,
                         is_user_correlations_cached=is_user_correlations_cached,
                         user_correlations=user_correlations,
                         min_common_elements=min_common_elements,
                         use_avg_ratings_cache=use_avg_ratings_cache)

        self.time_constraint = time_constraint
        self.use_bulk_corr_cache = use_bulk_corr_cache
        self.user_corrs_in_bulk = None

    def is_temporal_cache_valid(self) -> bool:
        """
        Check whether cache is valid or not.
        :return:
        """
        # No TimeConstraint, valid
        if self._time_constraint is None:
            return True
        # Bin TimeConstraint or Max Limit TimeConstraint, valid
        if self._time_constraint.is_valid_time_bin() or self._time_constraint.is_valid_max_limit():
            return True
        # Else, Not Valid
        return False

    def get_user_corrs_from_bulk(self, min_common_elements, time_constraint, bin_size):
        """
        Gets the user correlations if the time_constrain's matches from bulk cache(if found).
        :param min_common_elements: Min common elements used when calculating k_neighbours
        :param time_constraint: TimeConstraint to apply when calculating k neighbours
        :param bin_size: If the time_constraints is type of time_bin, used to get cache value.
        :return: User correlations from cache
        """
        if ((self.user_corrs_in_bulk is None) or (self.user_corrs_in_bulk is None)
                or (time_constraint is None) or self.min_common_elements != min_common_elements):
            return None

        if time_constraint.is_valid_max_limit():
            return self.user_corrs_in_bulk.get(time_constraint.end_dt.year)

        if bin_size == -1:
            return None

        bins = self.user_corrs_in_bulk.get(bin_size)
        if bins is not None:
            return bins.get(time_constraint.start_dt.year)

    def get_user_corrs(self, min_common_elements, time_constraint=None):
        """
        If user correlations cached returns the cache, else none

        :param min_common_elements: min common element in between users in order them to become neighbours
        :param time_constraint: time constraint on user correlations
        :return: user correlation matrix if cache found, else None
        """
        if self.is_user_correlations_cached:
            if self.time_constraint == time_constraint and self.min_common_elements == min_common_elements:
                return self.user_correlations
        return None

    def set_user_corrs(self, user_corrs, min_common_elements, time_constraint):
        """
        Update User correlations
        :param user_corrs: new user correlations
        :param min_common_elements: min common elements used when calculating 'user_corrs'
        :param time_constraint: time_constraint applied when calculating 'user_corrs'
        """

        # Only set when caching is open for user_correlations
        if self.is_user_correlations_cached:
            self._time_constraint = time_constraint
            self.min_common_elements = min_common_elements
            self.user_correlations = user_corrs

    @property
    def time_constraint(self):
        return self._time_constraint

    @time_constraint.setter
    def time_constraint(self, value):
        self._time_constraint = value
