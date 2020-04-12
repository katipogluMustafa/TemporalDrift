from .constraints import TimeConstraint
from .similarity import TemporalPearson
from .cache import TemporalCache, Cache
from datetime import datetime
import pandas as pd
import random


class TrainsetUser:

    def __init__(self, cache: Cache):
        """
        :param cache: Input cache must have movie_ratings not None !
        """
        self.cache = cache

        if not self.cache.is_movie_ratings_cached:
            raise Exception("'movie_ratings' has not been cached !")

    def get_users(self):
        """
        Get list of unique 'user_id's

        Since MovieLens Have 'user_id's from 0 to 610 without any missing user, for now sending that directly
        Uncomment the other lines later

        :return: the ids of the users found in movie_ratings
        """
        #
        # if self.cache.is_ratings_cached:
        #     data = self.cache.ratings
        # else:
        #     data = self.cache.movie_ratings
        #
        # return pd.unique(data['user_id'])
        return range(0, 611)

    def get_active_users(self, n=10) -> pd.DataFrame:
        """
        Get Users in sorted order where the first one is the one who has given most ratings.

        :param n: Number of users to retrieve.
        :return: user DataFrame with index of 'user_id' and columns of ['mean_rating', 'No_of_ratings'] .
        """

        if self.cache.is_ratings_cached:                         # 30% faster than other choice
            data = self.cache.ratings
        else:
            data = self.cache.movie_ratings

        active_users = pd.DataFrame(data.groupby('user_id')['rating'].mean())
        active_users['No_of_ratings'] = pd.DataFrame(data.groupby('user_id')['rating'].count())
        active_users.sort_values(by=['No_of_ratings'], ascending=False, inplace=True)
        active_users.columns = ['mean_rating', 'No_of_ratings']
        return active_users.head(n)

    def get_random_users(self, n=1):
        """
        Get list of random n number of 'user_id's

        :param n: Number of random users
        :return: List of random 'user_id's
        """

        return random.choices(population=self.get_users(), k=n)

    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """
        Get all the ratings given by of the chosen users

        :param user_id: id of the chosen user
        :return: Ratings given by the 'user_id'
        """
        if self.cache.is_ratings_cached:                         # 2.2x faster than other choice
            data = self.cache.ratings
        else:
            data = self.cache.movie_ratings

        return data.loc[data['user_id'] == user_id]

    def get_user_avg(self, user_id: int):

        if self.cache.use_avg_ratings_cache:
            avg_user_rating = self.cache.avg_user_ratings.loc[user_id]
            return avg_user_rating[0] if not avg_user_rating.empty else 0

        user_ratings = self.get_user_ratings(user_id=user_id)
        return user_ratings.rating.mean() if not user_ratings.empty else 0

    def get_timestamp(self, user_id: int, movie_id: int):
        """
        Get the timestamp of the given rating

        :param user_id: the users whose rating timestamp we are searching
        :param movie_id: id of the movie that the user gave the rating
        :return: if found the datetime object otherwise None
        """

        if self.cache.is_ratings_cached:
            data = self.cache.ratings
        else:
            data = self.cache.movie_ratings

        timestamp = data.loc[(data['user_id'] == user_id) & (data['item_id'] == movie_id)]
        return timestamp.values[0, 3] if not timestamp.empty else None

    def get_first_timestamp(self):
        if self.cache.is_ratings_cached:
            data = self.cache.ratings
        else:
            data = self.cache.movie_ratings
        return data['timestamp'].min()

    def get_user_avg_timestamp(self, user_id: int):
        user_ratings = self.get_user_ratings(user_id=user_id)
        return user_ratings.timestamp.mean() if not user_ratings.empty else 0

    # TODO: Later, create TemporalDatasetUser, and put this method into that one
    def get_user_ratings_at(self, user_id: int, at: datetime) -> pd.DataFrame:
        """
        Get user ratings up until the given datetime
        :param user_id: id of the chosen user
        :param at: only those ratings that are before this date will be taken into account
        :return: Ratings given by the 'user_id' before given datetime
        """

        if self.cache.is_ratings_cached:
            data = self.cache.ratings
        else:
            data = self.cache.movie_ratings

        return data.loc[(data['user_id'] == user_id) & (data.timestamp < at)]

    # TODO: Later, create TemporalDatasetUser, and put this method into that one
    def get_user_avg_at(self, user_id: int, at: datetime):
        user_ratings = self.get_user_ratings_at(user_id, at)
        return user_ratings.rating.mean() if not user_ratings.empty else 0


class TrainsetMovie:

    def __init__(self, cache: Cache):
        """
        :param cache: Input cache must have movie_ratings not None !
        """
        self.cache = cache

        if not self.cache.is_movie_ratings_cached:
            raise Exception("'movie_ratings' has not been cached !")

    def get_movie(self, movie_id) -> pd.DataFrame:
        """
        Get Movie Record

        :return: DataFrame which contains the given 'movie_id's details. If not found empty DataFrame .
        """
        if self.cache.is_movies_cached:
            return self.cache.movies.loc[self.cache.movies['item_id'] == movie_id]
        return self.cache.movie_ratings.loc[self.cache.movie_ratings['item_id'] == movie_id]

    def get_movies(self):
        """
        Get list of unique 'item_id's or in other words the movies.

        :return: List of movie ids
        """

        if self.cache.is_movies_cached:
            return self.cache.movies['item_id'].values.tolist()

        return pd.unique(self.cache.movie_ratings['item_id'])

    def get_random_movies(self, n=10):
        """
        Get list of random n number of 'item_id's or in other words the movies

        :param n: Number of random movies
        :return: List of random 'movie_id's
        """
        return random.choices(population=self.get_movies(), k=n)

    def get_movies_watched(self, user_id: int, time_constraint: TimeConstraint = None) -> pd.DataFrame:
        """
        Get all the movies watched by the chosen user.

        :param user_id: the user that we want to get the movies he-she has watched.
        :param time_constraint: type of the time constraint.
        :return: DataFrame of all movies watched with 'item_id', 'rating' columns
        """

        movie_ratings = self.cache.movie_ratings

        if time_constraint is None:
            return movie_ratings.loc[(movie_ratings['user_id'] == user_id)][['item_id', 'rating']]

        if time_constraint.is_valid_max_limit():
            return movie_ratings.loc[(movie_ratings['user_id'] == user_id)
                                     & (movie_ratings.timestamp < time_constraint.end_dt)][['item_id', 'rating']]
        elif time_constraint.is_valid_time_bin():
            return movie_ratings.loc[(movie_ratings['user_id'] == user_id)
                                     & (movie_ratings.timestamp >= time_constraint.start_dt)
                                     & (movie_ratings.timestamp < time_constraint.end_dt)][['item_id', 'rating']]
        raise Exception("Undefined time_constraint is given!")

    def get_movie_rating(self, movie_id: int, user_id: int) -> int:
        """
        Get the movie rating taken by the chosen user

        :param movie_id: the movie chosen movie's id
        :param user_id: id of the chosen user
        :return: Rating given by user. If not found, returns 0
        """

        if self.cache.is_ratings_cached:
            data = self.cache.ratings
        else:
            data = self.cache.movie_ratings

        movie_rating = data.loc[(data['user_id'] == user_id) & (data['item_id'] == movie_id)]
        return movie_rating.values[0, 2] if not movie_rating.empty else 0

    def get_random_movie_watched(self, user_id: int) -> int:
        """
        Get random movie id watched.

        :param user_id: User of interest
        :return:  movie_id or item_id of the random movie watched by the user.
                  In case non-valid user_id supplied then returns 0
        """
        movies_watched = self.get_movies_watched(user_id=user_id)
        return random.choice(movies_watched['item_id'].values.tolist()) if not movies_watched.empty else 0

    def get_random_movies_watched(self, user_id: int, n=2) -> pd.DataFrame:
        """
        Get random n movies watched by the user. Only use when n > 2

        Use get_random_movie_watched if n=1 since that one 2 fold faster.

        :param user_id: the user of interest
        :param n: number of random movies to get
        :return: DataFrame of movies, if none found then empty DataFrame
        """
        movies_watched = self.get_movies_watched(user_id=user_id)
        return random.choices(population=movies_watched['item_id'].values.tolist(),
                              k=n) if not movies_watched.empty else movies_watched

    def get_random_movie_per_user(self, user_id_list):
        """
        Get random movie for each user given in the 'user_id_list'

        :param user_id_list: List of valid user_ids
        :return: List of (user_id, movie_id) tuples
                where each movie_id is randomly chosen from watched movies of the user_id .
                In case any one of the user_id's supplies invalid, then the movie_id will be 0 for that user.
        """
        user_movie_list = list()
        for user_id in user_id_list:
            user_movie_list.append((user_id, self.get_random_movie_watched(user_id=user_id)))
        return user_movie_list


class Trainset:
    def __init__(self, cache: TemporalCache, min_common_elements: int = 5):
        self.cache = cache
        self.min_common_elements = min_common_elements
        self.similarity = TemporalPearson(time_constraint=None, cache=self.cache)

        if not self.cache.is_movie_ratings_cached:
            raise Exception("'movie_ratings' has not been cached !")

        self.trainset_movie = TrainsetMovie(cache=cache)
        self.trainset_user = TrainsetUser(cache=cache)

        # if caching is allowed, create user correlations cache
        self.similarity.get_corr_matrix()

    def predict_movies_watched(self, user_id, n=10, k=10, time_constraint=None) -> pd.DataFrame:
        """

        :param user_id: user of interest
        :param n: Number of movies to predict
        :param k: k neighbours to take into account
        :param time_constraint: When calculating k neighbours,
                                only those that comply to time_constraints will be taken into account.
        :return: DataFrame of Predictions where columns = ['prediction', 'rating'] index = 'movie_id'
        """
        # Get all movies watched by a user
        movies_watched = self.trainset_movie.get_movies_watched(user_id=user_id)

        if movies_watched.empty:
            return None

        predictions = list()
        number_of_predictions = 0
        for row in movies_watched.itertuples(index=False):
            prediction = self.predict_movie(user_id=user_id, movie_id=row[0],
                                            time_constraint=time_constraint, k=k)
            if number_of_predictions == n:
                break
            predictions.append([prediction, row[1], row[0]])
            number_of_predictions += 1

        predictions_df = pd.DataFrame(predictions, columns=['prediction', 'rating', 'movie_id'])
        predictions_df.movie_id = predictions_df.movie_id.astype(int)
        return predictions_df.set_index('movie_id')

    def predict_movie(self, user_id, movie_id, k=10, time_constraint=None, bin_size=-1):
        return self.similarity.mean_centered_pearson(user_id=user_id,
                                                     movie_id=movie_id,
                                                     k_neighbours=self.get_k_neighbours(user_id, k=k,
                                                                                        time_constraint=time_constraint,
                                                                                        bin_size=bin_size)
                                                     )

    def get_k_neighbours(self, user_id, k=20, time_constraint: TimeConstraint = None, bin_size=-1):
        """
        :param user_id: the user of interest
        :param k: number of neighbours to retrieve
        :param time_constraint: time constraint when choosing neighbours
        :param bin_size: Used when using time_bins, in order to select bin from cache
        :return: Returns the k neighbours and correlations in between them. If no neighbours found, returns None
                 DataFrame which has 'Correlation' column and 'user_id' index.
        """
        self.similarity.time_constraint = time_constraint
        user_corr_matrix = self.similarity.get_corr_matrix(bin_size=bin_size)

        # Exit if matrix is None, no user found in self.cache.movie_ratings, something is wrong
        if user_corr_matrix is None:
            return None

        # Get the chosen 'user_id's correlations
        user_correlations = user_corr_matrix.get(user_id)
        if user_correlations is None:
            return None

        # Drop any null, if found
        user_correlations.dropna(inplace=True)
        # Create A DataFrame from not-null correlations of the 'user_id'
        users_alike = pd.DataFrame(user_correlations)
        # Rename the only column to 'correlation'
        users_alike.columns = ['correlation']

        # Sort the user correlations in descending order
        #     so that first one is the most similar, last one least similar
        users_alike.sort_values(by='correlation', ascending=False, inplace=True)

        # Eliminate Correlation to itself by deleting first row,
        #     since biggest corr is with itself it is in first row
        return users_alike.iloc[1:k+1]


