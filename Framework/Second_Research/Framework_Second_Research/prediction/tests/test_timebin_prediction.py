import unittest

from internal.platform.accuracy.accuracy_metrics import Accuracy
from internal.platform.dataset_operators.dataset_user_operator import DatasetUserOperator
from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
from internal.platform.prediction.timebin_prediction import StaticTimebinPrediction


class TestTimebinPrediction(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestTimebinPrediction, self).__init__(*args, **kwargs)
    self.dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')
    self.optimized_dataset = DatasetOptimizer(self.dataset)
    self.dataset_user_operator = DatasetUserOperator(self.optimized_dataset.get_ratings())

  def test_prediction(self):
    static_timebin_prediction = StaticTimebinPrediction(self.optimized_dataset)
    static_timebin_prediction.predict(448, 3)


  @staticmethod
  def __calculate_first_derivatives(data):
    first_derivatives = [0] * len(data)
    for i in range(1, len(data)):
      first_derivatives[i] = data[i][1] - data[i-1][1]
    return first_derivatives

  @staticmethod
  def __calculate_second_derivatives(fist_derivatives):
    second_derivatives = [0] * len(fist_derivatives)
    for i in range(1, len(fist_derivatives)):
      second_derivatives[i] = fist_derivatives[i] - fist_derivatives[i - 1]
    return second_derivatives

  def test_dynamic_performance_using_derivatives(self):
    data = [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 2.25), (16, 0.25), (17, 0), (18, 0), (19, 0), (20, 0.25), (21, 0), (22, 0), (23, 0.25), (24, 0), (25, 0.25), (26, 0), (27, 0), (28, 0.25), (29, 0.25), (30, 0), (31, 1.0), (32, 2.25), (33, 0), (34, 0.001), (35, 0.25), (36, 0.25), (37, 2.25), (38, 0.25), (39, 0), (40, 2.25), (41, 0), (42, 2.25)]
    print(data)
    first_derivatives = [0] * len(data)
    for i in range(1, len(data)):
      first_derivatives[i] = data[i][1] - data[i-1][1]
    print(first_derivatives)

    second_derivatives = [0] * len(data)
    for i in range(1, len(data)):
      second_derivatives[i] = first_derivatives[i] - first_derivatives[i - 1]
    print(second_derivatives)

    for i in range(len(data)):
      print(f"{data[i]} {first_derivatives[i]} {second_derivatives[i]}")

  def test_best_dynamic_size(self):
    data = [(3, 0), (4, 0), (5, 0), (6, 0), (7, 0), (8, 0), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0),
            (15, 2.25), (16, 0.25), (17, 0), (18, 0), (19, 0), (20, 0.25), (21, 0), (22, 0), (23, 0.25), (24, 0),
            (25, 0.25), (26, 0), (27, 0), (28, 0.25), (29, 0.25), (30, 0), (31, 1.0), (32, 2.25), (33, 0), (34, 0.001),
            (35, 0.25), (36, 0.25), (37, 2.25), (38, 0.25), (39, 0), (40, 2.25), (41, 0), (42, 2.25)]
    non_zero_rmse_pairs = list()
    for i in range(len(data)):
      if 0.1 < data[i][1] < 1:
        non_zero_rmse_pairs.append(data[i])
    print(non_zero_rmse_pairs)
    # take inverse weighted average
    dynamic_timebin_size = 43
    product_sum = 0
    weights_sum = 0
    for i in range(len(non_zero_rmse_pairs)):
      weight = 1/non_zero_rmse_pairs[i][1]
      product_sum += non_zero_rmse_pairs[i][0] * weight
      weights_sum += weight
    dynamic_timebin_size = product_sum / weights_sum
    closest_timebin_size = -1
    closest_diff = float('inf')
    for i in range(len(non_zero_rmse_pairs)):
      diff = abs(non_zero_rmse_pairs[i][0] - dynamic_timebin_size)
      if diff < closest_diff:
        closest_timebin_size = non_zero_rmse_pairs[i][0]
        closest_diff = diff
    dynamic_timebin_size = closest_timebin_size if closest_timebin_size < dynamic_timebin_size else dynamic_timebin_size
    print(dynamic_timebin_size)

  def test_dynamic_timebin_size(self):
    user_id = 448
    movie_id = 3
    n_ratings = len(self.dataset_user_operator.get_user_rating_history(user_id))
    rmse_per_timebinsize = list()
    for timebin_size in range(3, 43, 1):
      static_timebin_prediction = StaticTimebinPrediction(self.optimized_dataset, global_timebin_size=timebin_size)
      predictions = list()
      for random_movie_id in self.dataset_user_operator.get_user_random_movie_list_from_history(user_id, n_movies=10):
        actual = self.dataset_user_operator.get_user_rating_value(user_id, random_movie_id)
        prediction = static_timebin_prediction.predict(user_id, random_movie_id)
        if prediction != 0:
          print(f"Movie_id: {random_movie_id} \tPrediction: {prediction} \tActual: {actual}")
          predictions.append( (prediction, actual) )
      rmse = Accuracy.rmse(predictions)
      rmse_per_timebinsize.append( (timebin_size, rmse) )

    print(rmse_per_timebinsize)

if __name__ == '__main__':
  unittest.main()
