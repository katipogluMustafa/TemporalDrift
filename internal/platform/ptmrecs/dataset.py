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
    def __init__(self,
                 ratings_col_names=('user_id', 'item_id', 'rating', 'timestamp'),
                 ratings_path=r'C:\Users\Yukawa\datasets\ml-latest-small\ratings.csv',
                 movies_col_names=('item_id', 'title', 'genres'),
                 movies_path=r'C:\Users\Yukawa\datasets\ml-latest-small\movies.csv',
                 is_ratings_cached=True,
                 is_movies_cached=True):
        Dataset.__init__(self)
        self.is_ratings_cached = is_ratings_cached
        self.is_movies_cached = is_movies_cached
        self.ratings = MovieLensDataset.load_ratings(ratings_path,
                                                     ratings_col_names) if self.is_ratings_cached else None
        self.movies = MovieLensDataset.load_movies(movies_path,
                                                   movies_col_names) if self.is_movies_cached else None

    @staticmethod
    def load_movies(movies_path,
                    movies_col_names=('item_id', 'title', 'genres')):
        if not os.path.isfile(movies_path) or not movies_col_names:
            return None

        # read movies
        movies = pd.read_csv(movies_path, sep=',', header=1, names=movies_col_names)

        # Extract Movie Year
        movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
        movies.year = pd.to_datetime(movies.year, format='%Y')
        movies.year = movies.year.dt.year  # As there are some NaN years, resulting type will be float (decimals)

        # Remove year part from the title
        movies.title = movies.title.str[:-7]

        return movies

    @staticmethod
    def load_ratings(ratings_path,
                     ratings_col_names=('user_id', 'item_id', 'rating', 'timestamp')):
        if not os.path.isfile(ratings_path) or not ratings_col_names:
            return None

        # read ratings
        ratings = pd.read_csv(ratings_path, sep=',', header=1, names=ratings_col_names)

        # Convert timestamp into readable format
        ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s', origin='unix')

        return ratings

    @staticmethod
    def create_movie_ratings(ratings, movies):
        return pd.merge(ratings, movies, on='item_id')

    @staticmethod
    def load(ratings_col_names=('user_id', 'item_id', 'rating', 'timestamp'),
             ratings_path=r'C:\Users\Yukawa\datasets\ml-latest-small\ratings.csv',
             movies_col_names=('item_id', 'title', 'genres'),
             movies_path=r'C:\Users\Yukawa\datasets\ml-latest-small\movies.csv'
             ):
        # Load movies
        movies = MovieLensDataset.load_movies(movies_path=movies_path, movies_col_names=movies_col_names)
        # Load ratings
        ratings = MovieLensDataset.load_ratings(ratings_path=ratings_path, ratings_col_names=ratings_col_names)

        # Merge the ratings and movies
        movie_ratings = pd.merge(ratings, movies, on='item_id')

        return movie_ratings


# #Unit Test
# movies_ratings = MovieLensDataset.load()
# print(movies_ratings)
