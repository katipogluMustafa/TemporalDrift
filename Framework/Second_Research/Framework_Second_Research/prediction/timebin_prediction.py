from internal.platform.dataset_operators.dataset_user_operator import DatasetUserOperator
from internal.platform.neighbour_filters.k_nearest_neighbourhood import KNearestNeighbours
from internal.platform.neighbour_filters.min_correlation_filter import MinCorrelationFilter
from internal.platform.neighbour_filters.significance_weighting_filter import SignificanceWeightingFilter
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
from internal.platform.prediction.prediction import Prediction
from internal.platform.similarity.timebin_similarity.timebin_similarity import TimebinSimilarity

class StaticTimebinPrediction(Prediction):
  def __init__(self, optimized_dataset: DatasetOptimizer, k=20, global_timebin_size=43,
               min_neighbour_timebin_size=5, max_neighbour_timebin_size=50, neighbour_timebin_size_increment=5,
               min_common_between_users=3):
    self.__timebin_similarity = TimebinSimilarity(optimized_dataset, min_neighbour_timebin_size,
                                                  max_neighbour_timebin_size,
                                                  neighbour_timebin_size_increment, min_common_between_users)
    super().__init__(self.__timebin_similarity, k)
    self.__optimized_dataset = optimized_dataset
    self.__dataset_user_operator = DatasetUserOperator(optimized_dataset.get_ratings())
    self.__k = k
    self.__global_timebin_size = global_timebin_size

  def predict(self, user_id, movie_id):
    neighbours = self.__timebin_similarity.get_neighbours(user_id, movie_id, self.__global_timebin_size)
    if neighbours.empty:
      return 0.0
    SignificanceWeightingFilter.filter(neighbours, correlation_column_name='pearson_corr', n_common_column_name='n_common')
    corr_filtered_neighbours = MinCorrelationFilter.filter(neighbours,
                                                           minimum_correlation=0.0,
                                                           correlation_column_name='pearson_corr')
    knn = KNearestNeighbours.get_k_nearest(corr_filtered_neighbours, self.__k, 'pearson_corr')
    print(knn)
    prediction = self.calculate_neighbour_weighted_avg_rating(movie_id, user_id, knn, corr_column_name='pearson_corr')
    actual = self.__dataset_user_operator.get_user_rating_value(user_id, movie_id)
    return prediction

class DynamicTimebinPrediction(Prediction):
  def __init__(self, optimized_dataset: DatasetOptimizer, k=20,
               min_neighbour_timebin_size=5, max_neighbour_timebin_size=50, neighbour_timebin_size_increment=5,
               min_common_between_users=3):
    self.__timebin_similarity = TimebinSimilarity(optimized_dataset, min_neighbour_timebin_size,
                                                  max_neighbour_timebin_size,
                                                  neighbour_timebin_size_increment, min_common_between_users)
    super().__init__(self.__timebin_similarity, k)
    self.__optimized_dataset = optimized_dataset
    self.__dataset_user_operator = DatasetUserOperator(optimized_dataset.get_ratings())
    self.__k = k

  def get_dynamic_timebin_size(self, user_id:int, movie_id:int):

    return 5

  def predict(self, user_id, movie_id):
    timebin_size = self.get_dynamic_timebin_size(user_id, movie_id)
    neighbours = self.__timebin_similarity.get_neighbours(user_id, movie_id, timebin_size)
    if neighbours.empty:
      return 0.0
    SignificanceWeightingFilter.filter(neighbours, correlation_column_name='pearson_corr', n_common_column_name='n_common')
    corr_filtered_neighbours = MinCorrelationFilter.filter(neighbours,
                                                           minimum_correlation=0.0,
                                                           correlation_column_name='pearson_corr')
    knn = KNearestNeighbours.get_k_nearest(corr_filtered_neighbours, 20, 'pearson_corr')
    print(knn)
    prediction = self.calculate_neighbour_weighted_avg_rating(movie_id, user_id, knn, corr_column_name='pearson_corr')
    actual = self.__dataset_user_operator.get_user_rating_value(user_id, movie_id)
    print(f"Prediction: {prediction} Actual: {actual}")

    return prediction