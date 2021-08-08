import os
import pandas as pd


class InvalidDatasetInputFilePath(Exception):
  pass


class Dataset:
  def __init__(self, ratings_file_path, movies_file_path):
    self.ratings_file_path = ratings_file_path
    self.movies_file_path = movies_file_path

    self.ratings_column_names = ('user_id', 'item_id', 'rating', 'timestamp')
    self.movies_column_names = ('item_id', 'title', 'genres')

  def load_movies(self) -> pd.DataFrame:
    return pd.DataFrame()

  def load_ratings(self) -> pd.DataFrame:
    return pd.DataFrame()

  def load_movie_ratings(self) -> pd.DataFrame:
    movies = self.load_movies()
    ratings = self.load_ratings()
    movie_ratings = Dataset.merge_ratings_and_movies_to_movie_ratings(ratings, movies)
    return movie_ratings

  @staticmethod
  def merge_ratings_and_movies_to_movie_ratings(ratings, movies) -> pd.DataFrame:
    return pd.merge(ratings, movies, on='item_id')

  @staticmethod
  def is_valid_input_file(path):
    return os.path.isfile(path)

  @staticmethod
  def sort_ratings_by_timestamp(ratings):
    ratings.sort_values(by='timestamp', inplace=True)