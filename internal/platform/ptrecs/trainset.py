from dataset import MovieLensDataset
import pandas as pd
from datetime import datetime


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



