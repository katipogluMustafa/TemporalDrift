from dataset import MovieLensDataset
from timeit import default_timer              # used in unit tests, see commented lines at the end
from datetime import datetime
from constraints import TimeConstraint
import pandas as pd
import random


class Trainset:
    def __init__(self, movie_ratings: pd.DataFrame):
        self.movie_ratings = movie_ratings

    # TODO: NAME CHANGED
    def get_user_ratings(self, user_id: int) -> pd.DataFrame:
        """
        Get all the ratings given by of the chosen users

        :param user_id: id of the chosen user
        :return: Ratings given by the 'user_id'
        """
        return self._movie_ratings.loc[self._movie_ratings['user_id'] == user_id]

    def get_user_ratings_at(self, user_id: int, at: datetime) -> pd.DataFrame:
        """
        Get user ratings up until the given datetime
        :param user_id: id of the chosen user
        :param at: only those ratings that are before this date will be taken into account
        :return: Ratings given by the 'user_id' before given datetime
        """
        return self.movie_ratings.loc[(self.movie_ratings['user_id'] == user_id)
                                      & (self.movie_ratings.timestamp < at)]

    # TODO: NAME CHANGED
    def get_movie_rating(self, movie_id: int, user_id: int) -> int:
        """
        Get the movie rating taken by the chosen user

        :param movie_id: the movie chosen movie's id
        :param user_id: id of the chosen user
        :return: Rating given by user. If not found, returns 0
        """
        movie_rating = self._movie_ratings.loc[(self._movie_ratings['user_id'] == user_id)
                                               & (self._movie_ratings['item_id'] == movie_id)]
        return movie_rating.values[0, 2] if not movie_rating.empty else 0

    # TODO: NAME CHANGED
    # TODO: Check None
    def get_timestamp(self, user_id: int, movie_id: int):
        """
        Get the timestamp of the given rating

        :param user_id: the users whose rating timestamp we are searching
        :param movie_id: id of the movie that the user gave the rating
        :return: if found the datetime object otherwise None
        """
        timestamp = self._movie_ratings.loc[(self._movie_ratings['user_id'] == user_id)
                                            & (self._movie_ratings['item_id'] == movie_id)]
        return timestamp.values[0, 3] if not timestamp.empty else None

    # TODO: NAME CHANGED
    def get_user_avg(self, user_id: int):
        user_ratings = self.get_user_ratings(user_id=user_id)
        return user_ratings.rating.mean() if not user_ratings.empty else 0

    def get_user_avg_timestamp(self, user_id: int):
        user_ratings = self.get_user_ratings(user_id=user_id)
        return user_ratings.timestamp.mean() if not user_ratings.empty else 0

    # TODO: NAME CHANGED
    def get_user_avg_at(self, user_id: int, at: datetime):
        user_ratings = self.get_user_ratings_at(user_id, at)
        return user_ratings.rating.mean() if not user_ratings.empty else 0

    def get_movies_watched(self, user_id: int, time_constraint: TimeConstraint = None) -> pd.DataFrame:
        """
        Get the movies watched by the chosen user.

        :param user_id: the user that we want to get the movies he-she has watched.
        :param time_constraint: type of the time constraint.
        :return: DataFrame of all movies watched with 'item_id', 'rating' columns
        """
        if time_constraint is None:
            return self._movie_ratings.loc[(self._movie_ratings['user_id'] == user_id)][['item_id', 'rating']]

        if time_constraint.is_valid_max_limit():
            return self._movie_ratings.loc[(self._movie_ratings['user_id'] == user_id)
                                           & (self._movie_ratings.timestamp < time_constraint.end_dt)][['item_id', 'rating']]
        elif time_constraint.is_valid_time_bin():
            print("Valid Bin\n")
            return self._movie_ratings.loc[(self._movie_ratings['user_id'] == user_id)
                                           & (self._movie_ratings.timestamp >= time_constraint.start_dt)
                                           & (self._movie_ratings.timestamp < time_constraint.end_dt)][['item_id', 'rating']]
        raise Exception("Undefined time_constraint is given!")

    # TODO: relocate this method to another class. it doesnt feel right here
    # TODO: NAME CHANGED
    # TODO: check if empty at the caller
    def get_random_movies_watched(self, user_id: int, n=2) -> pd.DataFrame:
        """
        Get random n movies watched by the user. Only use when n > 2

        Use get_random_movie_watched if n=1 since that one 2 fold faster.

        :param user_id: the user of interest
        :param n: number of random movies to get
        :return: DataFrame of movies, if none found then empty DataFrame
        """
        movies_watched = self.get_movies_watched(user_id=user_id)
        return movies_watched.sample(n=n) if not movies_watched.empty else movies_watched

    # TODO: check if 0, which means user does not exists on the caller
    def get_random_movie_watched(self, user_id: int) -> int:
        """
        Get random movie id watched.

        :param user_id: User of interest
        :return:  movie_id or item_id of the random movie watched by the user.
                  In case non-valid user_id supplied then returns 0
        """
        movies_watched = self.get_movies_watched(user_id=user_id)
        return random.choice(movies_watched['item_id'].values.tolist()) if not movies_watched.empty else 0

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

    def get_active_users(self, n=10) -> pd.DataFrame:
        """
        Get Users in sorted order where the first one is the one who has given most ratings.
        :param n: Number of users to retrieve.
        :return: user DataFrame with index of 'user_id' and columns of ['mean_rating', 'No_of_ratings'] .
        """
        active_users = pd.DataFrame(self._movie_ratings.groupby('user_id')['rating'].mean())
        active_users['No_of_ratings'] = pd.DataFrame(self._movie_ratings.groupby('user_id')['rating'].count())
        active_users.sort_values(by=['No_of_ratings'], ascending=False)
        active_users.columns = ['mean_rating', 'No_of_ratings']
        return active_users.head(n)

    # TODO: NAME CHANGED
    # TODO: This function have high cost, Look for alternative
    # TODO: Think to store user_id and movie_id range in Dataset and answer from there
    def get_users(self):
        """
        Get list of unique 'user_id's

        :return: the ids of the users found in movie_ratings
        """
        return pd.unique(self._movie_ratings['user_id'])

    # TODO: Make low cost version, choosing from unique values have high cost
    def get_random_users(self, n=1):
        """
        Get list of random n number of 'user_id's

        :param n: Number of random users
        :return: List of random 'user_id's
        """

        return random.choices(population=self.get_users(), k=n)

    # TODO: NAME CHANGED
    # TODO: This function have high cost, Look for alternative
    # TODO: Think to store user_id and movie_id range in Dataset and answer from there
    def get_movies(self):
        """
        Get list of unique 'item_id's or in other words the movies.

        :return: List of movie ids
        """
        return pd.unique(self._movie_ratings['item_id'])

    # TODO: Make low cost version, choosing from unique values have high cost
    def get_random_movies(self, n=10):
        """
        Get list of random n number of 'item_id's or in other words the movies

        :param n: Number of random movies
        :return: List of random 'movie_id's
        """
        return random.choices(population=self.get_movies(), k=n)

    # Properties
    @property
    def movie_ratings(self):
        return self._movie_ratings

    @movie_ratings.setter
    def movie_ratings(self, value):
        self._movie_ratings = value


# Unit Tests
t = Trainset(MovieLensDataset.load())
# print(t.movie_ratings)
# print(t.get_user_ratings(700))
# print(t.get_movie_rating(527, 3))
# print(t.get_timestamp(655,3))
# print(t.get_user_avg(3))
# print(t.get_user_avg_rating_at(3, datetime(2019,3,5)))
# print(t.get_random_movies_watched(user_id=3))
# print(t.get_random_users())

# st = default_timer()
# x = t.get_random_movies_watched(user_id=3)
# print(f"time taken = {default_timer() - st}, movie= {x}")
#
# st = default_timer()
# x = t.get_random_movie_watched(user_id=3)
# print(f"time taken = {default_timer() - st}, movie= {x}")
#
# print(t.get_active_users(5))

# print(t.get_user_avg_timestamp(548))
# print(t.get_movies_watched(548, time_constraint=TimeConstraint(end_dt=datetime(2017,3,28), start_dt=datetime(2017,3,3))))




