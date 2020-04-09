from abc import ABC, abstractmethod
import pandas as pd
import os


class Dataset(ABC):
    @staticmethod
    @abstractmethod
    def load():
        """ Every subclass must provide static load method"""
        pass


class MovieLensDataset(Dataset):
    def __init__(self):
        Dataset.__init__(self)

    @staticmethod
    def load(ratings_col_names=('user_id', 'item_id', 'rating', 'timestamp'),
             ratings_path=r'C:\Users\Yukawa\datasets\ml-latest-small\ratings.csv',
             movies_col_names=('item_id', 'title', 'genres'),
             movies_path=r'C:\Users\Yukawa\datasets\ml-latest-small\movies.csv'
             ):

        # Check if given paths are valid
        if not os.path.isfile(ratings_path) or not os.path.isfile(movies_path):
            return None

        # Check given column names are not empty
        if not ratings_col_names or not movies_col_names:
            return None

        # Read the files
        ratings = pd.read_csv(ratings_path, sep=',', header=1, names=ratings_col_names)
        movies = pd.read_csv(movies_path, sep=',', header=0, names=movies_col_names)

        # Extract Movie Year
        movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
        movies.year = pd.to_datetime(movies.year, format='%Y')
        movies.year = movies.year.dt.year  # As there are some NaN years, resulting type will be float (decimals)
        # Remove year part from the title
        movies.title = movies.title.str[:-7]
        # Convert timestamp into readable format
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s', origin='unix')

        # Merge the ratings and movies
        movie_ratings = pd.merge(ratings, movies, on='item_id')

        return movie_ratings


# #Unit Test
# movies_ratings = MovieLensDataset.load()
# print(movies_ratings)