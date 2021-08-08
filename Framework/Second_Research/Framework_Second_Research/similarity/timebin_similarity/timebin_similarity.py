import math
from collections import defaultdict

import pandas as pd
from internal.platform.dataset_operators.dataset_user_operator import DatasetUserOperator
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
from internal.platform.similarity.timebin_similarity.timebin import Timebin


class TimebinSimilarity:

  def __init__(self, optimized_dataset: DatasetOptimizer,
               neighbour_min_timebin_size=5,
               neighbour_max_timebin_size=50,
               neighbour_timebin_size_increment=1,
               min_n_common_between_users=3):
    self.optimized_dataset = optimized_dataset
    self.__dataset_user_operator = DatasetUserOperator(self.optimized_dataset.get_ratings())
    self.__neighbour_min_timebin_size = neighbour_min_timebin_size
    self.__neighbour_max_timebin_size = neighbour_max_timebin_size
    self.__neighbour_timebin_size_increment = neighbour_timebin_size_increment
    self.__min_n_common_between_neighbour_users = min_n_common_between_users

  def get_neighbours(self, user_id, movie_id, target_timebin_size=43):
    user_history = self.__dataset_user_operator.get_user_rating_history(user_id)
    target_movie_index = Timebin.find_movie_index_in_user_history(user_history, movie_id)
    if target_movie_index < 0:
      return pd.DataFrame()
    elif target_movie_index > target_timebin_size:
      timebin_size = target_timebin_size
      timebin_starting_index = target_movie_index - timebin_size
    elif 0 < target_movie_index < len(user_history):
      timebin_size = target_movie_index + 1
      timebin_starting_index = 0
    else:
      return pd.DataFrame()
    timebin = Timebin(self.__dataset_user_operator, user_id, timebin_starting_index, timebin_size)
    return self.get_timebin_neighbours(timebin, movie_id)

  def get_timebin_neighbours(self, timebin, movie_id):
    if timebin.get_timebin_df().empty:
      return pd.DataFrame()
    neighbours = self.get_qualified_neighbour_id_list(timebin, movie_id)
    neighbour_timebin_list = self.get_neighbour_timebin_list(movie_id, neighbours, timebin)
    similar_timebins = self.__drop_duplicate_neighbour_timebins_and_get_actual_similar_timebins(neighbour_timebin_list)
    return similar_timebins

  def get_qualified_neighbour_id_list(self, timebin_to_find_its_neighbours: Timebin, movie_id: int) -> list:
    timebin_user = timebin_to_find_its_neighbours.get_timebin_user()
    n_common_with_users_dict = self.__get_n_common_with_other_users(timebin_to_find_its_neighbours, movie_id)
    neighbour_id_list = self.__get_users_with_target_movie_rating_and_has_enough_history(movie_id, timebin_user,
                                                                                         n_common_with_users_dict)
    return neighbour_id_list

  def get_all_timebins_with_target_movie(self, user_id: int, movie_id: int) -> list:
    user_history = self.__dataset_user_operator.get_user_rating_history(user_id)
    neighbour_timebins = list()
    for timebin_size in range(self.__neighbour_min_timebin_size,
                              self.__neighbour_max_timebin_size,
                              self.__neighbour_timebin_size_increment):
      timebins = self.__generate_timebins_for_given_size(user_id, user_history, movie_id, timebin_size)
      neighbour_timebins.extend(timebins)
    return neighbour_timebins

  def get_neighbour_timebin_list(self, movie_id, neighbours, timebin):
    neighbour_timebin_list = list()
    for neighbour_id in neighbours:
      neighbour_timebins = self.get_all_timebins_with_target_movie(neighbour_id, movie_id)
      valid_neighbour_timebins = self.__get_valid_neighbour_timebins(neighbour_timebins, timebin)
      neighbour_timebin_list.extend(valid_neighbour_timebins)
    return neighbour_timebin_list

  @staticmethod
  def __is_invalid_correlation(correlation):
    return math.isnan(correlation)

  def __get_users_with_target_movie_rating_and_has_enough_history(self, movie_id, user_id, n_common_with_users):
    neighbour_id_list = []
    for possible_neighbour_id, n_rating in n_common_with_users.items():
      if n_rating > self.__min_n_common_between_neighbour_users:
        if self.__dataset_user_operator.get_user_rating_value(possible_neighbour_id, movie_id) != 0:
          if possible_neighbour_id != user_id:
            neighbour_id_list.append(possible_neighbour_id)
    return neighbour_id_list

  def __get_n_common_with_other_users(self, timebin: Timebin, movie_id: int):
    timebin_df = timebin.get_timebin_df_without_target_movie(movie_id)
    ratings = self.optimized_dataset.get_ratings().reset_index()
    n_common_with_users = defaultdict(int)
    for curr_movie_id in timebin_df.index.values.tolist():
      movie_raters = ratings.loc[(ratings['item_id'] == curr_movie_id)][['user_id']].values.tolist()
      for rater in movie_raters:
        n_common_with_users[rater[0]] += 1
    return n_common_with_users

  @staticmethod
  def __drop_duplicate_neighbour_timebins_and_get_actual_similar_timebins(data):
    similar_timebins = pd.DataFrame(data,
                                    columns=['neighbour_id', 'n_common', 'pearson_corr', 'timebin_i', 'timebin_size'])
    similar_timebins.drop_duplicates(['neighbour_id', 'n_common', 'pearson_corr', 'timebin_i'], inplace=True)
    return similar_timebins

  def __get_valid_neighbour_timebins(self, neighbour_timebins, timebin: Timebin):
    valid_neighbour_timebins = list()
    for neighbour_timebin in neighbour_timebins:
      corr, common_elements = timebin.get_correlation_and_n_common_between_timebins(neighbour_timebin)
      if not self.__is_invalid_correlation(corr) and self.__has_enough_common_elements(common_elements):
        neighbour_timebin_range = neighbour_timebin.get_timebin_range()
        valid_neighbour_timebins.append((neighbour_timebin.get_timebin_user(), common_elements, corr,
                                         neighbour_timebin_range[0], neighbour_timebin_range[1]))
    return valid_neighbour_timebins

  @staticmethod
  def __has_enough_common_elements(common_elements):
    return common_elements > 2

  def __generate_timebins_for_given_size(self, user_id, user_history, movie_id, timebin_size):
    neighbour_timebins = list()
    n_movies = len(user_history)
    for i in range(0, n_movies, timebin_size):
      timebin = Timebin(self.__dataset_user_operator, user_id, i, timebin_size)
      timebin_df = timebin.get_timebin_df(user_history)
      if movie_id in timebin_df.index:
        neighbour_timebins.append(timebin)
    return neighbour_timebins

  def get_dataset_optimizer(self):
    return self.optimized_dataset
