import math

from internal.platform.dataset_operators.dataset_user_operator import DatasetUserOperator
import pandas as pd
import numpy as np


class Timebin:
  def __init__(self, dataset_user_operator: DatasetUserOperator, user_id: int, timebin_starting_index, timebin_size):
    self.__dataset_user_operator = dataset_user_operator
    self.__user_id = user_id
    self.__timebin_starting_index = timebin_starting_index
    self.__timebin_size = timebin_size
    self.__cached_timebin_df = None
    self.__timebin_user_avg_rating = None

  def get_timebin_df(self, user_rating_history=None) -> pd.DataFrame:
    if self.__cached_timebin_df is not None:
      return self.__cached_timebin_df
    if user_rating_history is None or user_rating_history.empty:
      user_rating_history = self.__dataset_user_operator.get_user_rating_history(self.__user_id)
    timebin_end_index = self.__timebin_starting_index + self.__timebin_size
    timebin_df = self.get_timebin_from_user_history(user_rating_history, self.__timebin_starting_index,
                                                    timebin_end_index)
    self.__cached_timebin_df = timebin_df
    return timebin_df

  def get_timebin_df_without_target_movie(self, movie_id, user_rating_history=None):
    timebin = self.get_timebin_df(user_rating_history)
    timebin.drop(movie_id, inplace=True, errors='ignore')
    return timebin

  @staticmethod
  def find_timebin_starting_index_with_movie(rating_history, movie_id, timebin_size):
    timebin_last_index = Timebin.find_movie_index_in_user_history(rating_history, movie_id)
    return timebin_last_index - timebin_size

  @staticmethod
  def find_movie_index_in_user_history(ratings_history, movie_id):
    ratings_history.reset_index(inplace=True)
    timebin_last_index = np.where(ratings_history["item_id"] == movie_id)[0][0]
    return timebin_last_index

  def get_timebin_range(self):
    return self.__timebin_starting_index, self.__timebin_size

  def get_timebin_user(self):
    return self.__user_id

  def get_timebin_user_avg_rating(self):
    if self.__timebin_user_avg_rating is not None:
      return self.__timebin_user_avg_rating
    self.__timebin_user_avg_rating = self.__dataset_user_operator.get_user_avg(self.get_timebin_user())
    return self.__timebin_user_avg_rating

  @staticmethod
  def get_timebin_from_user_history(user_rating_history, timebin_start_i, timebin_end_i):
    if user_rating_history.empty:
      return pd.DataFrame()
    return user_rating_history.iloc[timebin_start_i:timebin_end_i]

  def get_correlation_and_n_common_between_timebins(self, timebin_b) -> (float, int):
    timebin_a_df = self.get_timebin_df()
    timebin_b_df = timebin_b.get_timebin_df()
    timebin_a_and_b_merged = self.__get_merged_timebin_with_common_movies(timebin_a_df, timebin_b_df)
    pearson = self.__calculate_pearson_correlation_between_timebins(timebin_a_and_b_merged,
                                                                    self.get_timebin_user_avg_rating(),
                                                                    timebin_b.get_timebin_user_avg_rating())
    return pearson, len(timebin_a_and_b_merged)

  @staticmethod
  def __get_merged_timebin_with_common_movies(timebin_a, timebin_b):
    return timebin_a.merge(timebin_b, on='item_id')

  @staticmethod
  def __calculate_pearson_correlation_between_timebins(merged, timebin_a_user_avg, timebin_b_user_avg):
    numerator = ((merged['rating_x'] - timebin_a_user_avg) * (merged['rating_y'] - timebin_b_user_avg)).sum()
    denominator = math.sqrt(((merged['rating_x'] - timebin_a_user_avg) ** 2).sum())
    denominator *= math.sqrt(((merged['rating_y'] - timebin_b_user_avg) ** 2).sum())
    pearson = numerator / denominator if denominator != 0 else 0
    return pearson
