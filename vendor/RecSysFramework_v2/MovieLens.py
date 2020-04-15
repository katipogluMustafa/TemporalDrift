# -*- coding: utf-8 -*-
"""
Created on Thu May    3 10:22:34 2018
Updated on Tue March  4 07:27:25 2020

@Authors: Frank , Mustafa Katipoglu
@Style: Google Python Style
"""

import os
import csv
import sys
import re

from surprise import Dataset
from surprise import Reader

from collections import defaultdict

class MovieLens:
    """ MovieLens class helps to retrieve miscellaneous data from MovieLens dataset """
    movieID_to_name = {}
    name_to_movieID = {}

    ratings_path = r'../../../data/datasets/ml-latest-small/ratings.csv'
    movies_path = r'../../../data/datasets/ml-latest-small/movies.csv'

    def load_movielens_latest_small(self):
        """ Load MovieLens-Small Dataset into MovieLens object"""
        # Set relative path to look for directories
        os.chdir(os.path.dirname(sys.argv[0]))

        ratings_dataset = 0
        self.movieID_to_name = {}
        self.name_to_movieID = {}

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        ratings_dataset = Dataset.load_from_file(self.ratings_path, reader=reader)

        with open(self.movies_path, newline='', encoding='ISO-8859-1') as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)   # Skip header line
            for row in movie_reader:
                movie_id = int(row[0])
                movie_name = row[1]
                self.movieID_to_name[movie_id] = movie_name
                self.name_to_movieID[movie_name] = movie_id

        return ratings_dataset

    def get_user_ratings(self,user):
        """ Returns user-rating lists"""
        user_ratings = []
        hit_user = False
        with open(self.ratings_path, newline='') as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)  # Skip header line
            for row in rating_reader:
                user_id = int(row[0])
                if user == user_id :
                    movie_id = int(row[1])
                    rating = float(row[2])
                    user_ratings.append((movie_id,rating))
                    hit_user = True
                if hit_user and (user != user_id) :           #TODO: Reevaluate the second condition
                    break

        return user_ratings

    def get_popularity_ranks(self):
        """
        Calculates popularity ranks
        :return: A default dict of ranks user_id -> rank
        """
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratings_path, newline='') as csvfile:
            rating_reader = csv.reader(csvfile)
            next(rating_reader)   # Skip header line
            for row in rating_reader:
                movie_id = int(row[1])
                ratings[movie_id] +=1
        rank = 1
        for movie_id, rating_count in sorted(ratings.items(),key=lambda x: x[1], reverse=True):
            rankings[movie_id] = rank
            rank += 1

        return rankings

    def get_genres(self):
        """
            Returns a default dict
            movie_id -> [genre list, 1 if true, 0 if not]
            related genres marked as 1 in the list of the movie_id
        """
        genres = defaultdict(list)
        genre_id_dict = {}
        max_genre_id = 0
        with open(self.movies_path, newline='', encoding='ISO-8859-1') as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)       # Skip header line
            for row in movie_reader:
                movie_id = int(row[0])
                genre_list = row[2].split('|')
                genre_id_list = []
                for genre in genre_list:
                    if genre in genre_id_dict:
                        genre_id = genre_id_dict[genre]
                    else:
                        genre_id = max_genre_id
                        genre_id_dict[genre] = genre_id
                        max_genre_id += 1
                    genre_id_list.append(genre_id)
                genres[movie_id] = genre_id_list

        # Convert integer-encoded genre lists to bitfields that we can treat as vectors
        for (movie_id, genre_id_list) in genres.items():
            bitfield = [0] * max_genre_id
            for genre_id in genre_id_list:
                bitfield[genre_id] = 1
            genres[movie_id] = bitfield

        return genres

    def get_timestamps(self):
        """
            Returns time stamps as a defaultdict
            movie_id -> timestamp
        """
        p = re.compile(r"(?:\((\d{4})\))?\s*$")
        timestamps = defaultdict(int)
        with open(self.movies_path, newline='', encoding='ISO-8859-1') as csvfile:
            movie_reader = csv.reader(csvfile)
            next(movie_reader)      # Skip header line
            for row in movie_reader:
                movie_id = int(row[0])
                title = row[1]
                m = p.search(title)
                timestamp = m.group(1)
                if timestamp:
                    timestamps[movie_id] = int(timestamp)
        return timestamps

    def get_movie_name(self, movie_id):
        if movie_id in self.movieID_to_name:
            return self.movieID_to_name[movie_id]
        else:
            return ""

    def get_movie_id(self, movie_name):
        if movie_name in self.name_to_movieID:
            return self.name_to_movieID[movie_name]
        else:
            return 0