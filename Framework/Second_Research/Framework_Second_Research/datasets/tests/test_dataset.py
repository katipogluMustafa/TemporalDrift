import unittest
from internal.platform.datasets.dataset import Dataset
from internal.platform.datasets.movielens_dataset import MovielensDataset
from internal.platform.datasets.netflix_dataset import NetflixDataset


class TestNetflixDataset(unittest.TestCase):
  def test_dataset_loading(self):
    dataset = NetflixDataset(ratings_file_path=r'C:\Users\Yukawa\datasets\netflix\combined_data_1.txt',
                             movies_file_path=r'C:\Users\Yukawa\datasets\netflix\movie_titles.csv')

    movies = dataset.load_movies()
    self.assertEqual(len(movies) > 0, True)

    ratings = dataset.load_ratings()
    self.assertEqual(len(ratings) > 0, True)

    movie_ratings = dataset.load_movie_ratings()
    self.assertEqual(len(movie_ratings) > 0, True)

    print(movie_ratings.head(5))



class TestMovielensDataset(unittest.TestCase):
  def test_dataset_loading(self):
    dataset = MovielensDataset(
            ratings_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                              r'\ratings.csv',
            movies_file_path=r'C:\Users\Yukawa\PycharmProjects\ProjectAlpha\data\movie_datasets\ml-latest-small'
                             r'\movies.csv')

    movies = dataset.load_movies()
    self.assertEqual(len(movies) > 0, True)

    ratings = dataset.load_ratings()
    self.assertEqual(len(ratings) > 0, True)

    movie_ratings = dataset.load_movie_ratings()
    self.assertEqual(len(movie_ratings) > 0, True)

    movie_ratings2 = Dataset.merge_ratings_and_movies_to_movie_ratings(ratings, movies)
    self.assertEqual(len(movie_ratings2) > 0, True)


if __name__ == '__main__':
  unittest.main()
