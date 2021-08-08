import pandas as pd

from internal.platform.dataset_operators.dataset_user_operator import DatasetUserOperator


class SignificanceWeighting:

  def __init__(self, actual_similarity_method, correlation_column_name='correlation'):
    self.__similarity_method = actual_similarity_method
    self.__correlation_column_name = correlation_column_name
    self.__dataset_optimizer = actual_similarity_method.get_dataset_optimizer()
    self.__dataset_user_operator = DatasetUserOperator(self.__dataset_optimizer.get_ratings())

  def get_neighbours_using_common_rated_item_count(self, user_id: int, movie_id: int) -> pd.DataFrame:
    movie_n_common_based_neighbours = self.__get_common_movie_based_neighbours(user_id, movie_id)
    if movie_n_common_based_neighbours.empty:
      return pd.DataFrame()
    neighbour_correlations = list()
    for neighbour_id, row in movie_n_common_based_neighbours.iterrows():
      u = row['n_common']
      neighbour_correlations.append((neighbour_id, u * row[self.__correlation_column_name]))
    neighbours = self.__get_neighbours_df(neighbour_correlations)
    return neighbours

  def get_neighbours_using_static_significance_weighting(self, user_id: int, movie_id: int, alpha=50) -> pd.DataFrame:
    movie_n_common_based_neighbours = self.__get_common_movie_based_neighbours(user_id, movie_id)
    if movie_n_common_based_neighbours.empty:
      return pd.DataFrame()
    neighbour_correlations = list()
    for neighbour_id, row in movie_n_common_based_neighbours.iterrows():
      u = 1
      if row['n_common'] < alpha:
        u = row['n_common'] / alpha
      neighbour_correlations.append((neighbour_id, u * row[self.__correlation_column_name]))
    neighbours = self.__get_neighbours_df(neighbour_correlations)
    return neighbours

  def get_neighbours_using_dynamic_significance_weighting(self, user_id: int, movie_id: int,
                                                          beta=3 / 4) -> pd.DataFrame:
    movie_n_common_based_neighbours = self.__get_common_movie_based_neighbours(user_id, movie_id)
    if movie_n_common_based_neighbours.empty:
      return pd.DataFrame()
    mean_n_common = movie_n_common_based_neighbours['n_common'].mean()
    alpha = 2 * beta * mean_n_common
    neighbour_correlations = list()
    for neighbour_id, row in movie_n_common_based_neighbours.iterrows():
      u = 1
      if row['n_common'] < alpha:
        u = row['n_common'] / alpha
      neighbour_correlations.append((neighbour_id, u * row[self.__correlation_column_name]))
    neighbours = self.__get_neighbours_df(neighbour_correlations)
    return neighbours

  def __get_common_movie_based_neighbours(self, user_id: int, movie_id: int) -> pd.DataFrame:
    user_neighbors = self.__similarity_method.get_neighbours(user_id)
    if user_neighbors.empty:
      return pd.DataFrame()
    user_neighbors.sort_values(by=self.__correlation_column_name, ascending=False, inplace=True)
    movie_based_neighbours = self.__get_movie_based_neighbours(movie_id, user_id, user_neighbors)
    if movie_based_neighbours.empty:
      return pd.DataFrame()
    self.__refactor_neighbour_df_column_and_index(movie_based_neighbours)
    return movie_based_neighbours

  def __get_neighbours_df(self, neighbour_list: list) -> pd.DataFrame:
    neighbour_df = pd.DataFrame(neighbour_list)
    neighbour_df.columns = ['user_id', self.__correlation_column_name]
    neighbour_df.set_index('user_id', inplace=True)
    return neighbour_df

  def __refactor_neighbour_df_column_and_index(self, movie_based_neighbours):
    movie_based_neighbours.columns = ['user_id', self.__correlation_column_name, 'n_common']
    movie_based_neighbours.set_index('user_id', inplace=True)

  def __get_movie_based_neighbours(self, movie_id, user_id, user_knn) -> pd.DataFrame:
    user_neighbour_corr_and_n_common_list = list()
    target_user_hist = self.__dataset_user_operator.get_user_rating_history(user_id)
    for neighbour_id, row in user_knn.iterrows():
      rating = self.__dataset_user_operator.get_user_rating_value(neighbour_id, movie_id)
      neighbour_history = self.__dataset_user_operator.get_user_rating_history(neighbour_id)
      n_common = len(target_user_hist.merge(neighbour_history, on='item_id'))
      if rating == 0:
        continue
      user_neighbour_corr_and_n_common_list.append((neighbour_id, row['correlation'], n_common))
    return pd.DataFrame(user_neighbour_corr_and_n_common_list)
