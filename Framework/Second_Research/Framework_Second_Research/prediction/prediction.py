from internal.platform.dataset_operators.dataset_user_operator import DatasetUserOperator
from internal.platform.neighbour_filters.k_nearest_neighbourhood import KNearestNeighbours


class Prediction:
  def __init__(self, similarity_method, k: int = 10):
    self.__similarity_method = similarity_method
    self.__dataset_optimizer = self.__similarity_method.get_dataset_optimizer()
    self.__dataset_user_operator = DatasetUserOperator(self.__dataset_optimizer.get_ratings())
    self.__k = k

  def predict(self, user_id: int, movie_id: int) -> float:
    if self.__dataset_user_operator.get_user_rating_record(user_id, movie_id).empty:
      return 0.0
    target_user_k_nearest_neighbors = KNearestNeighbours(self.__similarity_method, self.__k)
    user_k_nearest_neighbours = target_user_k_nearest_neighbors.get_k_nearest_neighbours(user_id)
    if user_k_nearest_neighbours.empty:
      return 0.0
    return self.calculate_neighbour_weighted_avg_rating(movie_id, user_id, user_k_nearest_neighbours)

  def predict_using_given_neighbours(self, user_id: int, movie_id: int, neighbours,
                                     neighbours_corr_column_name='correlation') -> float:
    user_k_nearest_neighbours = KNearestNeighbours.get_k_nearest(neighbours, self.__k, neighbours_corr_column_name)
    if user_k_nearest_neighbours.empty:
      return 0.0
    return self.calculate_neighbour_weighted_avg_rating(movie_id, user_id, user_k_nearest_neighbours)

  def calculate_neighbour_weighted_avg_rating(self, movie_id, user_id, user_k_nearest_neighbours, corr_column_name='correlation') -> float:
    avg_user_rating = self.__dataset_user_operator.get_user_avg(user_id)
    sum_of_weights, weighted_sum = self.__take_weighted_neighbour_rating_average(movie_id, user_k_nearest_neighbours, corr_column_name)
    return avg_user_rating + (weighted_sum / sum_of_weights) if sum_of_weights != 0 else 0

  def __take_weighted_neighbour_rating_average(self, movie_id, user_k_nearest_neighbours, corr_column_name) -> (float, float):
    weighted_sum, sum_of_weights = 0.0, 0.0
    for neighbour_id, neighbour in user_k_nearest_neighbours.iterrows():
      neighbour_rating = self.__dataset_user_operator.get_user_rating_value(neighbour_id, movie_id)
      if neighbour_rating != 0:
        neighbour_zero_centered_rating = neighbour_rating - self.__dataset_user_operator.get_user_avg(neighbour_id)
        weighted_sum += neighbour_zero_centered_rating * neighbour[corr_column_name]
        sum_of_weights += neighbour[corr_column_name]
    return sum_of_weights, weighted_sum
