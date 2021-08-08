import unittest

from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.neighbour_filters.k_nearest_neighbourhood import KNearestNeighbours
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
from internal.platform.similarity.pearson import PearsonSimilarity
from internal.platform.similarity.significance_weighting import SignificanceWeighting


class TestSignificanceWeighting(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestSignificanceWeighting, self).__init__(*args, **kwargs)
    self.dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')
    self.optimized_dataset = DatasetOptimizer(self.dataset)
    self.pearson_similarity = PearsonSimilarity(self.optimized_dataset, 3)
    self.significance_weighting = SignificanceWeighting(self.pearson_similarity)

  def test_common_rated_item_count(self):
    self.__assert_common_rated_significance_weighting_exists(3, 10, 448)
    self.__assert_common_rated_significance_weighting_exists(3, 10, 440)

  def test_dynamic_significance_weighting(self):
    self.__assert_dynamic_significance_weighting_exists(3, 10, 448)
    self.__assert_dynamic_significance_weighting_exists(3, 10, 440)

  def test_static_significance_weighting(self):
    self.__assert_static_significance_weighting_exists(3, 10, 448)
    self.__assert_static_significance_weighting_exists(3, 10, 440)

  def __assert_static_significance_weighting_exists(self, item_id, k, user_id):
    neighbours = self.significance_weighting.get_neighbours_using_static_significance_weighting(user_id, item_id)
    knn = KNearestNeighbours.get_k_nearest(neighbours, k)
    self.assertTrue(len(knn) > 0)

  def __assert_dynamic_significance_weighting_exists(self, item_id, k, user_id):
    neighbours = self.significance_weighting.get_neighbours_using_dynamic_significance_weighting(user_id, item_id)
    knn = KNearestNeighbours.get_k_nearest(neighbours, k)
    self.assertTrue(len(knn) > 0)

  def __assert_common_rated_significance_weighting_exists(self, item_id, k, user_id):
    neighbours = self.significance_weighting.get_neighbours_using_common_rated_item_count(user_id, item_id)
    knn = KNearestNeighbours.get_k_nearest(neighbours, k)
    self.assertTrue(len(knn) > 0)


if __name__ == '__main__':
  unittest.main()
