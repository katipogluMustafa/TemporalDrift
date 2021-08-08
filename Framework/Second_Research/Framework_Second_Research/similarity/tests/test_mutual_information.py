import unittest

from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
from internal.platform.similarity.mutual_information import MutualInformation


class TestMutualInformation(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestMutualInformation, self).__init__(*args, **kwargs)
    self.dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')
    self.optimized_dataset = DatasetOptimizer(self.dataset)

  def test_something(self):
    mutual_info = MutualInformation(self.optimized_dataset)
    self.assertTrue(len(mutual_info.get_neighbours(448, 3)) > 0)


if __name__ == '__main__':
  unittest.main()
