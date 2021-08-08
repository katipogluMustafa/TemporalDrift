from internal.platform.datasets.dataset import Dataset, InvalidDatasetInputFilePath
import pandas as pd

class MovielensDataset(Dataset):

  def __init__(self, ratings_file_path, movies_file_path):
    super().__init__(ratings_file_path, movies_file_path)
    self.ratings_column_names = ('user_id', 'item_id', 'rating', 'timestamp')
    self.movies_column_names = ('item_id', 'title', 'genres')
    self.__lowest_rating, self.__highest_rating, self.__rating_increment = 0.5, 5, 0.5

  def get_dataset_rating_range(self):
    return self.__lowest_rating, self.__highest_rating, self.__rating_increment

  def load_movies(self) -> pd.DataFrame:
    if not Dataset.is_valid_input_file(self.movies_file_path):
      raise InvalidDatasetInputFilePath
    movies = MovielensDataset.__read_data_from_file(self.movies_file_path, self.movies_column_names, )
    MovielensDataset.__create_datetime_year_column_for_movies_from_title_years(movies)
    MovielensDataset.__truncate_movie_year_str_from_movie_titles(movies)
    return movies

  def load_ratings(self) -> pd.DataFrame:
    if not Dataset.is_valid_input_file(self.ratings_file_path):
      raise InvalidDatasetInputFilePath
    ratings = MovielensDataset.__read_data_from_file(self.ratings_file_path, self.ratings_column_names)
    MovielensDataset.__convert_ratings_timestamp_column_to_readable_dates(ratings)
    Dataset.sort_ratings_by_timestamp(ratings)
    return ratings

  @staticmethod
  def __read_data_from_file(file_path, column_names):
    return pd.read_csv(file_path, sep=',', header=1, names=column_names).set_index('item_id')

  @staticmethod
  def __create_datetime_year_column_for_movies_from_title_years(movies: pd.DataFrame):
    movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
    movies.year = pd.to_datetime(movies.year, format='%Y')
    movies.year = movies.year.dt.year  # As there are some NaN years, resulting type will be float (decimals)

  @staticmethod
  def __truncate_movie_year_str_from_movie_titles(movies: pd.DataFrame):
    movies.title = movies.title.str[:-7]

  @staticmethod
  def __convert_ratings_timestamp_column_to_readable_dates(ratings: pd.DataFrame):
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s', origin='unix')