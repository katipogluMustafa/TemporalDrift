import unittest
from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.datasets.dataset import Dataset
from internal.platform.dataset_operators.dataset_user_operator import DatasetUserOperator
import pandas as pd
from internal.platform.constraints.interval import *
from datetime import datetime


class TestDatasetUserOperator(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestDatasetUserOperator, self).__init__(*args, **kwargs)
    self.dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')
    self.ratings = self.dataset.load_ratings()
    self.movies = self.dataset.load_movies()
    self.movie_ratings = self.dataset.merge_ratings_and_movies_to_movie_ratings(self.ratings, self.movies)

    self.user_operator = DatasetUserOperator(self.ratings)

  def test_get_all_users(self):
    users = self.user_operator.get_all_users()
    self.assertEqual(len(users) > 0, True)

  def test_top_raters(self):
    raters = self.user_operator.get_top_n_raters(3)
    self.assertEqual(len(raters), 3)
    user_ratings_length = len(self.__get_target_user_ratings(TestDatasetUserOperator.__get_first_rater_user_id(raters)))
    self.assertEqual(user_ratings_length == TestDatasetUserOperator.__get_first_rater_rating_count(raters), True)

    raters = self.user_operator.get_top_n_raters(-1)
    self.assertEqual(len(raters), 0)

  def test_random_users(self):
    self.assertEqual(len(self.user_operator.get_random_user_list(2)), 2)
    self.assertLessEqual(len(self.user_operator.get_random_user_list(50000)), 50000)
    self.assertEqual(len(self.user_operator.get_random_user_list(0)), 0)
    self.assertEqual(len(self.user_operator.get_random_user_list(-5)), 0)

  def test_user_history(self):
    self.assertEqual(len(self.user_operator.get_user_rating_history(-1)) >= 0, True)
    self.assertEqual(len(self.user_operator.get_user_rating_history(0)) >= 0, True)
    self.assertEqual(len(self.user_operator.get_user_rating_history(414)) >= 1, True)
    self.assertEqual(len(self.user_operator.get_user_rating_history(99999999)) == 0, True)

  # Use this as get_movie_rating
  def test_user_rating(self):
    self.assertEqual(self.user_operator.get_user_rating_record(414, 839)['rating'], 4.0)
    self.assertEqual(self.user_operator.get_user_rating_record(414, -1).empty, True)
    self.assertEqual(self.user_operator.get_user_rating_record(-5, -1).empty, True)
    self.assertEqual(self.user_operator.get_user_rating_record(9999999, 5).empty, True)
    self.assertEqual(self.user_operator.get_user_rating_record(414, 999999).empty, True)
    self.assertEqual(self.user_operator.get_user_rating_record(3, 39).empty, True)

  def test_rating_timestamp(self):
    self.assertTrue(self.user_operator.get_rating_timestamp(414, 839) is not None)
    self.assertTrue(self.user_operator.get_rating_timestamp(3, 39) is None)
    self.assertTrue(self.user_operator.get_rating_timestamp(99999999, 39) is None)
    self.assertTrue(self.user_operator.get_rating_timestamp(-1, 39) is None)
    self.assertTrue(self.user_operator.get_rating_timestamp(414, -5) is None)

  def test_user_avg_timestamp(self):
    self.assertEqual(self.user_operator.get_user_avg_rating_timestamp(448) is not None, True)
    self.assertEqual(self.user_operator.get_user_avg_rating_timestamp(-1) is None, True)
    self.assertEqual(self.user_operator.get_user_avg_rating_timestamp(0) is None, True)
    self.assertEqual(self.user_operator.get_user_avg_rating_timestamp(9999999999) is None, True)

  def test_get_user_ratings_at_interval(self):
    self.assert_user_ratings_with_max_limit_within_interval(448, datetime(2010, 5, 6))
    self.assert_user_ratings_with_max_limit_within_interval(448, datetime(2025, 5, 6))
    self.assert_user_ratings_with_max_limit_within_interval(448, datetime(1800, 5, 6))

  def test_get_user_avg_at_interval(self):
    print(self.user_operator.get_user_avg_at_interval(448, MaxLimitInterval(interval_end_datetime=datetime(2015,5,5))))

  def assert_user_ratings_with_max_limit_within_interval(self, user_id, dt):
    last_timestamp = self.__get_last_timestamp_of_user_ratings_with_max_limit_interval(dt, user_id)
    if last_timestamp is not None:
      self.assertTrue(self.__get_last_timestamp_of_user_ratings_with_max_limit_interval(dt, user_id) <= dt)

  def __get_last_timestamp_of_user_ratings_with_max_limit_interval(self, dt, user_id):
    user_ratings = self.user_operator.get_user_ratings_at_interval(user_id, MaxLimitInterval(interval_end_datetime=dt))
    try:
      return TestDatasetUserOperator.__get_last_rating_timestamp_from_ratings(user_ratings)
    except IndexError:
      return None

  @staticmethod
  def __get_last_rating_timestamp_from_ratings(user_ratings):
    return user_ratings.iloc[-1]['timestamp']

  @staticmethod
  def __get_first_rating_timestamp_from_ratings(user_ratings):
    return user_ratings.iloc[0]['timestamp']

  def __get_target_user_ratings(self, user_id):
    return self.ratings.loc[self.ratings['user_id'] == user_id]

  @staticmethod
  def __get_first_rater_rating_count(raters):
    return raters.iloc[0][1]

  @staticmethod
  def __get_first_rater_user_id(raters):
    return raters.index[0]


if __name__ == '__main__':
  unittest.main()
