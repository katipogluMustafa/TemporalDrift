from constraints import TimeConstraint


class Cache:

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

        # if use avg ratings cache, on average 10 fold performance gain
        self.use_avg_ratings_cache = use_avg_ratings_cache
        if self.use_avg_ratings_cache:
            self.avg_user_ratings = self.create_user_avg_rating_cache()
        else:
            self.avg_user_ratings = None

    def create_user_avg_rating_cache(self):
        if self.is_ratings_cached:
            data = self.ratings
        else:
            data = self.movie_ratings
        return data.groupby('user_id')[['rating']].mean()

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


class TemporalCache(Cache):

    def __init__(self, time_constraint: TimeConstraint,
                 is_ratings_cached=False,
                 ratings=None,
                 is_movies_cached=False,
                 movies=None,
                 is_movie_ratings_cached=False,
                 movie_ratings=None,
                 is_user_movie_matrix_cached=False,
                 user_movie_matrix=None,
                 is_user_correlations_cached=False,
                 user_correlations=None
                 ):
        super().__init__(
                 is_ratings_cached=is_ratings_cached,
                 ratings=ratings,
                 is_movies_cached=is_movies_cached,
                 movies=movies,
                 is_movie_ratings_cached=is_movie_ratings_cached,
                 movie_ratings=movie_ratings,
                 is_user_movie_matrix_cached=is_user_movie_matrix_cached,
                 user_movie_matrix=user_movie_matrix,
                 is_user_correlations_cached=is_user_correlations_cached,
                 user_correlations=user_correlations)
        self.time_constraint = time_constraint

    def is_temporal_cache_valid(self):
        if self.time_constraint is None:
            return False
        if self.time_constraint.is_valid_time_bin() or self.time_constraint.is_valid_max_limit():
            return True
        return False

    @property
    def time_constraint(self):
        return self._time_constraint

    @time_constraint.setter
    def time_constraint(self, value):
        self._time_constraint = value





