import pandas as pd
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer


class PearsonSimilarity:
  def __init__(self, dataset_optimizer: DatasetOptimizer, min_common_elements: int):
    self.dataset_optimizer = dataset_optimizer
    self.min_common_elements = min_common_elements

  def get_neighbours(self, user_id: int, user_user_correlation_matrix=None) -> pd.DataFrame:
    if user_user_correlation_matrix is None:
      user_user_correlation_matrix = self.get_user_user_correlation_matrix()
    target_user_correlations = user_user_correlation_matrix.get(user_id)
    if target_user_correlations is None:
      raise TargetUserNotFoundException
    return self.__get_clean_target_user_correlations_df(target_user_correlations)

  def __get_clean_target_user_correlations_df(self, target_user_correlations):
    self.__drop_invalid_values(target_user_correlations)
    target_user_correlations_df = pd.DataFrame(target_user_correlations)
    target_user_correlations_df.columns = ['correlation']
    return target_user_correlations_df

  def get_dataset_optimizer(self):
    return self.dataset_optimizer

  @staticmethod
  def __drop_invalid_values(target_user_correlations):
    target_user_correlations.dropna(inplace=True)

  def get_user_user_correlation_matrix(self):
    movie_ratings = self.dataset_optimizer.get_movie_ratings()
    user_movie_matrix = movie_ratings.pivot_table(index='title', columns='user_id', values='rating')
    user_user_corr_matrix = user_movie_matrix.corr(method="pearson", min_periods=self.min_common_elements)
    return user_user_corr_matrix


class TargetUserNotFoundException(Exception):
  pass
