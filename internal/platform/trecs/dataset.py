import pandas as pd

class Dataset:
    """
        One should call load_ratings in order ratings to be loaded into the Dataset
        Use UserRating class to get anything about dataset
    """
    def __init__(self):
        self.ratings = None

    def load_ratings(self, filepath:str = r"../../../data/datasets/ml-latest-small/ratings.csv"):
        self.ratings = Dataset.get_ratings_from_file(filepath)
        return self

    @staticmethod
    def load_df_from_file(filepath:str) -> pd.DataFrame:
        return pd.read_csv(filepath, sep=',', header=1, names=('user_id', 'item_id', 'rating', 'timestamp'))

    @staticmethod
    def get_ratings_from_df(df:pd.DataFrame) -> (int, int, float, int):
        """ Return ratings as tuples of (user_id, item_id, rating, timestamp)"""
        return [ (int(user_id), int(item_id), float(rating), int(timestamp))
                 for (user_id, item_id, rating, timestamp) in df.itertuples(index=False)]

    @staticmethod
    def get_ratings_from_file(filepath):
        return Dataset.get_ratings_from_df(Dataset.load_df_from_file(filepath))
