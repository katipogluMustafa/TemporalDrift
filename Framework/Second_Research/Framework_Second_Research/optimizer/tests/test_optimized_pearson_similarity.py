import unittest

import pandas as pd

from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
from internal.platform.optimizer.pearson_optimizer import OptimizedPearsonSimilarity
from internal.platform.similarity.pearson import TargetUserNotFoundException


class TestOptimizedPearsonSimilarity(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestOptimizedPearsonSimilarity, self).__init__(*args, **kwargs)
    self.dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')
    self.optimized_dataset = DatasetOptimizer(self.dataset)
    self.optimized_pearson_similarity = OptimizedPearsonSimilarity(self.optimized_dataset, 3)

  def test_neighbours(self):
    self.assertTrue(len(self.optimized_pearson_similarity.get_neighbours(448)) > 0, True)
    self.assertRaises(TargetUserNotFoundException, self.optimized_pearson_similarity.get_neighbours, 888)

  def test_output_types(self):
    self.assertIsInstance(self.optimized_pearson_similarity.is_optimizer_active(), bool)
    self.assertIsInstance(self.optimized_pearson_similarity.get_neighbours(448), pd.DataFrame)
    self.assertIsInstance(self.optimized_pearson_similarity.get_user_user_correlation_matrix(), pd.DataFrame)


if __name__ == '__main__':
  unittest.main()
