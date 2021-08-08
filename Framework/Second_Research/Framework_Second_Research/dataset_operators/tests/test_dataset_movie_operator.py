import unittest

from internal.platform.dataset_operators.dataset_movie_operator import DatasetMovieOperator
from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer


class TestDatasetMovieOperator(unittest.TestCase):
  def test_get_movie_record(self):
    dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')
    movie_operator = DatasetMovieOperator(DatasetOptimizer(dataset));
    self.assertTrue(movie_operator.get_movie_record(9999999999).empty)
    self.assertFalse(movie_operator.get_movie_record(3).empty)
    self.assertTrue(movie_operator.get_movie_record(-1).empty)


if __name__ == '__main__':
  unittest.main()
