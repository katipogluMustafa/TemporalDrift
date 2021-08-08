import unittest

from internal.platform.dataset_operators.dataset_user_operator import DatasetUserOperator
from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
from internal.platform.similarity.timebin_similarity.timebin_similarity import TimebinSimilarity, Timebin


class TestTimebinSimilarity(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestTimebinSimilarity, self).__init__(*args, **kwargs)
    self.dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')
    self.optimized_dataset = DatasetOptimizer(self.dataset)
    self.dataset_user_operator = DatasetUserOperator(self.optimized_dataset.get_ratings())

  def test_static_timebin_similarity(self):
    #print(self.optimized_dataset.get_ratings())
    ratings = self.optimized_dataset.get_ratings()
    #print(ratings.reset_index())
    static_timebin_similarity = TimebinSimilarity(self.optimized_dataset)
    timebin = Timebin(self.dataset_user_operator, 448, 0, 80)
    print(static_timebin_similarity.get_timebin_neighbours(timebin, 3))
    self.assertEqual(False, False)


if __name__ == '__main__':
  unittest.main()
