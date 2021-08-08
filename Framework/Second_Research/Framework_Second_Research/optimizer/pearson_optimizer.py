import pandas as pd
from internal.platform.similarity.pearson import PearsonSimilarity
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer


class OptimizedPearsonSimilarity:
  def __init__(self, dataset_optimizer: DatasetOptimizer, min_common_elements: int, is_active=True):
    self.__pearson_similarity = PearsonSimilarity(dataset_optimizer, min_common_elements)
    self.__user_user_correlation_matrix = pd.DataFrame()
    self.__is_active = is_active

  def get_user_user_correlation_matrix(self) -> pd.DataFrame:
    if not self.is_optimizer_active():
      return self.__pearson_similarity.get_user_user_correlation_matrix()
    if not self.__is_user_user_correlations_cached():
      self.__cache_user_correlations()
    return self.__get_cached_user_correlations()

  def get_neighbours(self, user_id: int) -> pd.DataFrame:
    return self.__pearson_similarity.get_neighbours(user_id, self.get_user_user_correlation_matrix())

  def get_dataset_optimizer(self):
    return self.__pearson_similarity.get_dataset_optimizer()

  def is_optimizer_active(self):
    return self.__is_active

  def __cache_user_correlations(self):
    self.__user_user_correlation_matrix = self.__pearson_similarity.get_user_user_correlation_matrix()

  def __get_cached_user_correlations(self):
    return self.__user_user_correlation_matrix

  def __is_user_user_correlations_cached(self):
    return not self.__user_user_correlation_matrix.empty

