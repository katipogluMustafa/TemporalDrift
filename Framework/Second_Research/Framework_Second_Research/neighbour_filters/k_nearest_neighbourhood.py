import pandas as pd


class KNearestNeighbours:
  def __init__(self, similarity_method, k: int, correlation_column_name='correlation'):
    self.__similarity_method = similarity_method
    self.__k = k
    self.__correlation_column_name = correlation_column_name

  def get_k_nearest_neighbours(self, user_id: int) -> pd.DataFrame:
    user_neighbors = self.__similarity_method.get_neighbours(user_id)
    if user_neighbors.empty:
      return pd.DataFrame()
    user_neighbors.sort_values(by=self.__correlation_column_name, ascending=False, inplace=True)
    user_k_nearest_neighbors = user_neighbors.iloc[1:self.__k + 1]
    return user_k_nearest_neighbors

  @staticmethod
  def get_k_nearest(neighbours: pd.DataFrame, k: int, correlation_column_name='correlation') -> pd.DataFrame:
    if neighbours is None or neighbours.empty:
      return pd.DataFrame()
    neighbours_df = neighbours.copy(deep=True)
    neighbours_df.sort_values(by=correlation_column_name, ascending=False, inplace=True)
    k_nearest_neighbours = neighbours_df.iloc[1:k + 1]
    return k_nearest_neighbours
