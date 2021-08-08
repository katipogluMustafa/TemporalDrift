from internal.platform.datasets.dataset import Dataset, InvalidDatasetInputFilePath
from collections import deque
import numpy as np
import pandas as pd


class NetflixDataset(Dataset):

  def __init__(self, ratings_file_path, movies_file_path):
    super().__init__(ratings_file_path, movies_file_path)
    self.ratings_column_names = ('user_id', 'rating', 'timestamp')
    self.movies_column_names = ('item_id', 'year', 'title')
    self.__lowest_rating, self.__highest_rating, self.__rating_increment = 1, 5, 1

  def get_dataset_rating_range(self):
    return self.__lowest_rating, self.__highest_rating, self.__rating_increment

  def load_movies(self):
    movies = self.__read_movies_data_from_file()
    movies = NetflixDataset.__replace_invalid_years_with_zero(movies)
    movies = NetflixDataset.__convert_movies_year_format_to_int(movies)
    movies = NetflixDataset.__interchange_movies_title_and_year(movies)
    return NetflixDataset.__get_only_movies_from_the_first_ratings_file(movies)

  def load_ratings(self):
    if not Dataset.is_valid_input_file(self.ratings_file_path):
      raise InvalidDatasetInputFilePath
    unstructured_ratings = self.__read_ratings_data_from_file()
    ratings = NetflixDataset.__structure_ratings_dataframe(unstructured_ratings)
    ratings = NetflixDataset.__reduce_dataset_size_by_removing_low_active_user_data(ratings)
    Dataset.sort_ratings_by_timestamp(ratings)
    return ratings

  @staticmethod
  def __get_only_movies_from_the_first_ratings_file(movies):
    return movies[:4499]

  @staticmethod
  def __interchange_movies_title_and_year(movies):
    return movies.reindex(columns=['title', 'year'])

  @staticmethod
  def __convert_movies_year_format_to_int(movies):
    movies['year'] = movies['year'].astype(int)
    return movies

  def __read_movies_data_from_file(self):
    return pd.read_csv(self.movies_file_path, encoding='ISO-8859-1', header=None,
                       names=self.movies_column_names).set_index('item_id')

  @staticmethod
  def __replace_invalid_years_with_zero(movies):
    movies['year'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    return movies

  def __read_ratings_data_from_file(self):
    return pd.read_csv(self.ratings_file_path, header=None, names=self.ratings_column_names, usecols=[0, 1, 2])

  @staticmethod
  def __structure_ratings_dataframe(ratings_raw):
    NetflixDataset.__convert_ratings_rating_column_format_to_float(ratings_raw)
    ratings = NetflixDataset.__convert_netflix_ratings_format_to_standard_movielens_like_format(ratings_raw)
    NetflixDataset.__convert_ratings_column_order_similar_to_movielens(ratings)
    NetflixDataset.__convert_ratings_timestamp_format_to_datetime(ratings)
    NetflixDataset.__convert_ratings_user_id_format_to_int(ratings)
    return ratings

  @staticmethod
  def __convert_ratings_rating_column_format_to_float(ratings):
    ratings['rating'] = ratings['rating'].astype(float)

  @staticmethod
  def __convert_ratings_user_id_format_to_int(ratings):
    ratings['user_id'] = ratings['user_id'].astype(int)

  @staticmethod
  def __convert_ratings_timestamp_format_to_datetime(ratings):
    ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], infer_datetime_format=True)

  @staticmethod
  def __convert_ratings_column_order_similar_to_movielens(ratings):
    ratings.reindex(columns=['user_id', 'item_id', 'rating', 'timestamp'])

  @staticmethod
  def __convert_netflix_ratings_format_to_standard_movielens_like_format(ratings_raw):

    # Find empty rows to slice dataframe for each movie
    temp_movies = ratings_raw[ratings_raw['rating'].isna()]['user_id'].reset_index()
    movie_indexes = [[index, int(movie[:-1])] for index, movie in temp_movies.values]
    # Shift the movie_indexes by one to get start and endpoints of all movies
    shifted_movie_indexes = deque(movie_indexes)
    shifted_movie_indexes.rotate(-1)
    # Gather all dataframes
    user_data = []
    for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indexes, shifted_movie_indexes):

      # if it is the last movie in the file
      if df_id_1 < df_id_2:
        temp_df = ratings_raw[(df_id_1 + 1):(df_id_2 - 1)].copy()
      else:
        temp_df = ratings_raw[df_id_1 + 1:].copy()

      # Create movie id column
      temp_df['item_id'] = movie_id

      # Append dataframe to list

      user_data.append(temp_df)
    # Combile all  dataframes
    ratings = pd.concat(user_data)
    del user_data, ratings_raw, temp_movies, temp_df, shifted_movie_indexes, movie_indexes, df_id_1, df_id_2, \
      movie_id, next_movie_id
    return ratings

  @staticmethod
  def __reduce_dataset_size_by_removing_low_active_user_data(ratings):
    netflix_users = NetflixDataset.__get_high_active_user_list(ratings, 40)
    ratings = NetflixDataset.__drop_low_active_user_ratings_only_keep_high_ones(netflix_users, ratings)
    return ratings

  @staticmethod
  def __get_high_active_user_list(ratings, min_ratings_count_for_being_high_active):
    data = ratings.copy(deep=True)
    users = pd.DataFrame(data.groupby('user_id')['rating'].mean())
    users['No_of_ratings'] = pd.DataFrame(data.groupby('user_id')['rating'].count())
    users.sort_values(by=['No_of_ratings'], ascending=False, inplace=True)
    users.columns = ['mean_rating', 'No_of_ratings']
    return users.loc[users['No_of_ratings'] > min_ratings_count_for_being_high_active].drop_duplicates(
      'mean_rating').drop_duplicates('No_of_ratings').index.values

  @staticmethod
  def __drop_low_active_user_ratings_only_keep_high_ones(netflix_users, ratings):
    return ratings.loc[(ratings['user_id'].isin(netflix_users))]