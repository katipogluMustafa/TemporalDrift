import unittest
from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
from internal.platform.optimizer.dataset_optimizer import InvalidDatasetOptimizerInput


class TestDatasetOptimizer(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestDatasetOptimizer, self).__init__(*args, **kwargs)
    self.dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')

  def test_load_ratings(self):
    movielens_dataset = self.dataset
    self.assertRaises(InvalidDatasetOptimizerInput, DatasetOptimizer, None)
    self.assert_valid_ratings(DatasetOptimizer(movielens_dataset))
    self.assert_valid_movies(DatasetOptimizer(movielens_dataset))

  def assert_valid_movies(self, optimized_dataset: DatasetOptimizer):
    self.assertTrue(len(optimized_dataset.get_movies()) > 0, True)

  def assert_valid_ratings(self, optimized_dataset: DatasetOptimizer):
    ratings = optimized_dataset.get_ratings()
    self.assertEqual(len(ratings) > 0, True)

  def test_clean_optimizer(self):
    dataset_optimizer = DatasetOptimizer(self.dataset)
    self.assertTrue(dataset_optimizer.__movies.empty, True)
    self.assertTrue(dataset_optimizer.__ratings.empty, True)
    self.assertTrue(dataset_optimizer.__movie_ratings.empty, True)

    dataset_optimizer.get_movies()
    dataset_optimizer.clean()
    self.assertTrue(dataset_optimizer.__movies.empty, True)

    dataset_optimizer.get_ratings()
    dataset_optimizer.clean()
    self.assertTrue(dataset_optimizer.__ratings.empty, True)

    dataset_optimizer.get_movie_ratings()
    dataset_optimizer.clean()
    self.assertTrue(dataset_optimizer.__movie_ratings.empty, True)


if __name__ == '__main__':
  unittest.main()
