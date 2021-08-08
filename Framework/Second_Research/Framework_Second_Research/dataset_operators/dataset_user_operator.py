import pandas as pd
from internal.platform.dataset_operators.dataset_operator import DatasetOperator
from internal.platform.constraints.interval import *
import numpy as np
import random


class DatasetUserOperator:

  def __init__(self, ratings: pd.DataFrame):
    self.__ratings = ratings

  def get_all_users(self) -> np.ndarray:
    return pd.unique(self.__ratings['user_id'])

  def get_top_n_raters(self, n) -> pd.DataFrame:
    if not DatasetUserOperator.__is_positive_number(n):
      return pd.DataFrame()
    active_users = self.__get_user_mean_ratings()
    active_users['count'] = self.__get_user_ratings_counts()
    DatasetUserOperator.__sort_active_ratings_by_count(active_users)
    DatasetUserOperator._rename_columns_as_mean_and_count(active_users)
    return active_users.head(n) if n is not None else active_users

  def get_random_user_list(self, n_users) -> list:
    if not DatasetUserOperator.__is_positive_number(n_users):
      return []
    return random.choices(population=self.get_all_users(), k=n_users)

  def get_user_rating_history(self, user_id: int) -> pd.DataFrame:
    if not DatasetUserOperator.__is_valid_user_id(user_id):
      return pd.DataFrame()
    return self.__ratings.loc[self.__ratings['user_id'] == user_id]

  def get_user_avg(self, user_id: int) -> int:
    user_ratings = self.get_user_rating_history(user_id)
    return user_ratings.rating.mean() if not user_ratings.empty else 0

  def get_user_rating_record(self, user_id: int, movie_id: int) -> pd.Series:
    if not DatasetUserOperator.__is_positive_number(movie_id) or not DatasetUserOperator.__is_valid_user_id(user_id):
      return pd.Series(dtype=object)
    try:
      return self.__get_target_user_movie_rating(user_id, movie_id)
    except:
      return pd.Series(dtype=object)

  def get_user_rating_value(self, user_id:int, movie_id:int):
    history = self.get_user_rating_record(user_id, movie_id)
    return history.values[0, 2] if not history.empty else 0

  def get_rating_timestamp(self, user_id: int, movie_id: int):
    user_rating = self.get_user_rating_record(user_id, movie_id)
    if user_rating.empty:
      return None
    return user_rating['timestamp']

  def get_user_avg_rating_timestamp(self, user_id: int):
    user_ratings = self.get_user_rating_history(user_id)
    return user_ratings.timestamp.mean() if not user_ratings.empty else None

  def get_user_ratings_at_interval(self, user_id: int, at: Interval) -> pd.DataFrame:
    ratings = DatasetOperator.apply_time_constraint(self.__ratings, at)
    return ratings.loc[(ratings['user_id'] == user_id)]

  def get_user_avg_at_interval(self, user_id: int, at: Interval):
    user_ratings = self.get_user_ratings_at_interval(user_id, at)
    return user_ratings.rating.mean() if not user_ratings.empty else 0

  def get_user_random_movie_from_history(self, user_id: int) -> int:
    rated_movie_ids = self.get_rated_movie_ids(user_id)
    if not rated_movie_ids:
      return 0
    return random.choice(rated_movie_ids)

  def get_user_random_movie_list_from_history(self, user_id:int, n_movies:int)->list:
    rated_movie_ids = self.get_rated_movie_ids(user_id)
    if not rated_movie_ids:
      return list()
    return random.choices(rated_movie_ids, k=n_movies)

  def get_rated_movie_ids(self, user_id:int):
    return self.get_user_rating_history(user_id).reset_index()['item_id'].values.tolist()

  @staticmethod
  def __is_positive_number(n_users):
    return n_users > 0

  @staticmethod
  def __is_valid_user_id(user_id):
    return user_id > 0

  @staticmethod
  def _rename_columns_as_mean_and_count(active_users):
    active_users.columns = ['mean', 'count']

  @staticmethod
  def __sort_active_ratings_by_count(active_users):
    active_users.sort_values(by=['count'], ascending=False, inplace=True)

  def __get_user_ratings_counts(self):
    return pd.DataFrame(self.__ratings.groupby('user_id')['rating'].count())

  def __get_user_mean_ratings(self):
    return pd.DataFrame(self.__ratings.groupby('user_id')['rating'].mean())

  def __get_target_user_movie_rating(self, user_id, movie_id):
    history = self.get_user_rating_history(user_id).reset_index()
    return history.loc[history['item_id'] == movie_id]
