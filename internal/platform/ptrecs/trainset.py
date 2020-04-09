from dataset import MovieLensDataset
import pandas as pd
from datetime import datetime
from constraints import PredictionTimeConstraint
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

    # TODO: NAME CHANGED
    def get_user_avg_at(self, user_id: int, at: datetime):
        user_ratings = self.get_user_ratings_at(user_id, at)
        return user_ratings.rating.mean() if not user_ratings.empty else 0

    def get_movies_watched(self, user_id: int,
                           time_constraint: PredictionTimeConstraint = None,
                           dt: datetime = None, start_dt: datetime = None, end_dt: datetime = None) -> pd.DataFrame:
        """
        Get the movies watched by the chosen user.

        :param user_id: the user that we want to get the movies he-she has watched.
        :param time_constraint: type of the time constraint.
        :param dt: used only when 'AT' time_constraint. Used as max date when getting movies
        :param start_dt: used only with 'IN' time_constraint.
               Used as start date of the interval to which we take the movies into account.
        :param end_dt: used only with 'IN' time_constraint. Used as the end date of the interval.
        :return: DataFrame of all movies watched with 'item_id', 'rating' columns
        """
        if time_constraint is None or time_constraint == PredictionTimeConstraint.NO:
            return self._movie_ratings.loc[(self._movie_ratings['user_id'] == user_id)][['item_id', 'rating']]
        elif time_constraint == PredictionTimeConstraint.AT and dt is not None:
            return self._movie_ratings.loc[(self._movie_ratings['user_id'] == user_id)
                                           & (self._movie_ratings.timestamp < dt)][['item_id', 'rating']]
        elif time_constraint == PredictionTimeConstraint.IN and start_dt is not None and end_dt is not None:
            return self._movie_ratings.loc[(self._movie_ratings['user_id'] == user_id)
                                           & (self._movie_ratings.timestamp >= start_dt)
                                           & (self._movie_ratings.timestamp < end_dt)][['item_id', 'rating']]
        else:
            raise Exception("Undefined time_constraint is given!")

    # TODO: relocate this method to another class. it doesnt feel right here
    # TODO: NAME CHANGED
    # TODO: check if empty at the caller
    def get_random_movies_watched(self, user_id, n=1) -> pd.DataFrame:
        """
        Get random n movies watched by the user

        :param user_id: the user of interest
        :param n: number of random movies to get
        :return: DataFrame of movies, if none found then empty DataFrame
        """
        movies_watched = self.get_movies_watched(user_id=user_id)
        return movies_watched.sample(n=n) if not movies_watched.empty else movies_watched

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


