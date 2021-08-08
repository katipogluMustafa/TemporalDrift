from internal.platform.datasets.dataset import Dataset
import pandas as pd

class DatasetOptimizer:
  def __init__(self, dataset:Dataset, is_active=True):
    if dataset is None:
      raise InvalidDatasetOptimizerInput
    self.__dataset = dataset
    self.__is_active = is_active
    self.__ratings = pd.DataFrame()
    self.__movies  = pd.DataFrame()
    self.__movie_ratings = pd.DataFrame()

  def get_ratings(self):
    if not self.is_optimizer_active():
      return self.__dataset.load_ratings()
    if self.__ratings.empty:
      self.__ratings = self.__dataset.load_ratings()
    return self.__ratings

  def get_movies(self):
    if not self.is_optimizer_active():
      return self.__dataset.load_movies()
    if self.__movies.empty:
      self.__movies = self.__dataset.load_movies()
    return self.__movies

  def get_movie_ratings(self):
    if not self.is_optimizer_active():
      return self.__dataset.load_movie_ratings()
    if self.__movie_ratings.empty:
      self.__movie_ratings = pd.merge(self.get_ratings(), self.get_movies(), on='item_id')
    return self.__movie_ratings

  def get_dataset(self):
    return self.__dataset

  def clean(self, clean_movies=True, clean_ratings=True, clean_movie_ratings=True):
    if clean_movies:
      self.__movies = pd.DataFrame()
    if clean_ratings:
      self.__ratings = pd.DataFrame()
    if clean_movie_ratings:
      self.__movie_ratings = pd.DataFrame()

  def is_optimizer_active(self):
    return self.__is_active

  def activate(self):
    self.__is_active = True

  def deactivate(self):
    self.__is_active = False

class InvalidDatasetOptimizerInput(Exception):
  pass

