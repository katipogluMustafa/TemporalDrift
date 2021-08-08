from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
import pandas as pd
import random

class DatasetMovieOperator:

  def __init__(self, dataset_optimizer:DatasetOptimizer):
    self.dataset_optimizer = dataset_optimizer

  def get_movie_record(self, movie_id: int) -> pd.Series:
    movies = self.dataset_optimizer.get_movies()
    try:
      movie_record = movies.loc[movie_id]
    except:
      movie_record = pd.Series(dtype=object)
    return movie_record

  def get_movie_id_list(self) -> list:
    movies = self.dataset_optimizer.get_movies()
    return movies['item_id'].values.tolist()

  def get_random_movie_id_list(self, number_of_movie_ids=10)->list:
    return random.choices(population=self.get_movie_id_list(), k=number_of_movie_ids)