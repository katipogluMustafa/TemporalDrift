import unittest
from datetime import datetime
from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.constraints.interval import Interval, TimebinInterval, MaxLimitInterval
from internal.platform.dataset_operators.dataset_operator import DatasetOperator

class TestDatasetOperator(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestDatasetOperator, self).__init__(*args, **kwargs)
    self.dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')
    self.movie_ratings = self.dataset.load_movie_ratings()
    self.movie_ratings_length = len(self.movie_ratings)

  def assert_data_with_time_constraint_has_less_length_dataframe(self, interval: Interval, assertion):
    with__timebin_constraint = DatasetOperator.apply_time_constraint(self.movie_ratings, interval)
    self.assertEqual(len(with__timebin_constraint) < self.movie_ratings_length, assertion)

  def test_applying_timebin_time_constraint(self):
    movie_ratings = self.movie_ratings.copy(True)

    DatasetOperator.apply_time_constraint(movie_ratings, None)
    self.assertEqual(len(movie_ratings) > 0, True)

    interval = TimebinInterval(datetime(2000, 5, 5), datetime(2020, 5, 5))
    self.assert_data_with_time_constraint_has_less_length_dataframe(interval, True)

    interval = TimebinInterval(None, datetime(2020, 5, 5))
    self.assert_data_with_time_constraint_has_less_length_dataframe(interval, False)

    interval = TimebinInterval(None, None)
    self.assert_data_with_time_constraint_has_less_length_dataframe(interval, False)

    interval = TimebinInterval(datetime(2020, 5, 5), None)
    self.assert_data_with_time_constraint_has_less_length_dataframe(interval, False)

  def test_applying_max_limit_time_constraint(self):
    movie_ratings = self.movie_ratings.copy(True)

    DatasetOperator.apply_time_constraint(movie_ratings, None)
    self.assertEqual(len(movie_ratings) > 0, True)

    interval = MaxLimitInterval(None, datetime(2010, 5, 5))
    self.assert_data_with_time_constraint_has_less_length_dataframe(interval, True)

    interval = MaxLimitInterval(None, None)
    self.assert_data_with_time_constraint_has_less_length_dataframe(interval, False)

    interval = MaxLimitInterval(datetime(2010, 5, 5), None)
    self.assert_data_with_time_constraint_has_less_length_dataframe(interval, False)

    interval = MaxLimitInterval(datetime(2000, 1, 1), datetime(2010, 5, 5))
    self.assert_data_with_time_constraint_has_less_length_dataframe(interval, False)


if __name__ == '__main__':
  unittest.main()
