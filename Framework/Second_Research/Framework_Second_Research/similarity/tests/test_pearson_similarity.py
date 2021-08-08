import unittest
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.similarity.pearson import TargetUserNotFoundException, PearsonSimilarity


class TestPearsonSimilarity(unittest.TestCase):
  def __init__(self, *args, **kwargs):
    super(TestPearsonSimilarity, self).__init__(*args, **kwargs)
    self.dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')
    self.optimized_dataset = DatasetOptimizer(self.dataset)

  def test_neighbour_selection(self):
    pearson_similarity = PearsonSimilarity(self.optimized_dataset, 3)
    self.assertRaises(TargetUserNotFoundException, pearson_similarity.get_neighbours, 888)
    self.assertTrue(len(pearson_similarity.get_neighbours(448)) > 0, True)
    print(pearson_similarity.get_user_user_correlation_matrix().empty)


if __name__ == '__main__':
  unittest.main()
