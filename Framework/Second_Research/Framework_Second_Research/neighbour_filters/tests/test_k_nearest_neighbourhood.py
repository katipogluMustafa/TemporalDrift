import unittest

from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.neighbour_selection.k_nearest_neighbourhood import KNearestNeighbours
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
from internal.platform.optimizer.pearson_optimizer import OptimizedPearsonSimilarity


class TestKNearestNeighbourhood(unittest.TestCase):
  def test_movie_based_neighbourhood(self):
    dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')
    pearson_similarity = OptimizedPearsonSimilarity(DatasetOptimizer(dataset), 3)
    knn = KNearestNeighbours(pearson_similarity, 20)
    self.assertTrue(len(knn.get_common_movie_based_k_nearest_neighbours(448, 3)) > 0)

if __name__ == '__main__':
  unittest.main()
