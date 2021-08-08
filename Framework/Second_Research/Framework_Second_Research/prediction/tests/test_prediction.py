import unittest
from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
from internal.platform.optimizer.pearson_optimizer import OptimizedPearsonSimilarity

from internal.platform.prediction.prediction import Prediction
from internal.platform.similarity.mutual_information import MutualInformation
from internal.platform.similarity.significance_weighting import SignificanceWeighting


class TestPrediction(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestPrediction, self).__init__(*args, **kwargs)
    self.dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')
    self.optimized_dataset = DatasetOptimizer(self.dataset)
    self.pearson_similarity = OptimizedPearsonSimilarity(self.optimized_dataset, 3)

  def test_pearson_prediction(self):
    pearson_prediction = Prediction(self.pearson_similarity)
    user_id = 448
    item_id = 3
    self.assertNotEqual(pearson_prediction.predict(user_id, item_id), 0)

  def test_significance_weighting_prediction(self):
    significance_weighting = SignificanceWeighting(self.pearson_similarity)
    significance_weighting_based_prediction = Prediction(self.pearson_similarity)

    neighbours = significance_weighting.get_neighbours_using_static_significance_weighting(448, 3)
    prediction = significance_weighting_based_prediction.predict_using_given_neighbours(448, 3, neighbours)
    print(f"Static: {prediction}\n")
    self.assertNotEqual(prediction, 0)

    neighbours = significance_weighting.get_neighbours_using_dynamic_significance_weighting(448, 3)
    prediction = significance_weighting_based_prediction.predict_using_given_neighbours(448, 3, neighbours)
    print(f"Dynamic: {prediction}\n")
    self.assertNotEqual(prediction, 0)

    neighbours = significance_weighting.get_neighbours_using_common_rated_item_count(448, 3)
    prediction = significance_weighting_based_prediction.predict_using_given_neighbours(448, 3, neighbours)
    print(f"Common Rated Item Count: {prediction}\n")
    self.assertNotEqual(prediction, 0)

  def test_mutual_information_prediction(self):
    mutual_info = MutualInformation(self.optimized_dataset)
    mutual_info_based_prediction = Prediction(self.pearson_similarity)

    neighbours = mutual_info.get_neighbours(448, 3)
    prediction = mutual_info_based_prediction.predict_using_given_neighbours(448, 3, neighbours)
    print(f"Mutual Info Prediction: {prediction}\n")
    self.assertNotEqual(prediction, 0)


if __name__ == '__main__':
  unittest.main()
