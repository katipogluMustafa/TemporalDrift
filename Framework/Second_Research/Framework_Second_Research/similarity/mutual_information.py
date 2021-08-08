import math

from internal.platform.dataset_operators.dataset_user_operator import DatasetUserOperator
from internal.platform.optimizer.dataset_optimizer import DatasetOptimizer
import pandas as pd


class MutualInformation:
  def __init__(self, dataset_optimizer: DatasetOptimizer):
    self.dataset_optimizer = dataset_optimizer
    dataset = self.dataset_optimizer.get_dataset()
    self.lowest_rating, self.highest_rating, self.rating_increment = dataset.get_dataset_rating_range()
    self.dataset_user_operator = DatasetUserOperator(self.dataset_optimizer.get_ratings())

  def get_neighbours(self, user_id, movie_id):
    neighbour_mutual_info = list()
    for dataset_user in self.dataset_user_operator.get_all_users():
      rating = self.dataset_user_operator.get_user_rating_value(dataset_user, movie_id)
      if rating != 0:
        corr = self.mutual_information(user_id, dataset_user)
        if corr != 0:
          # print(f"rating={rating} corr={corr}")
          neighbour_mutual_info.append((dataset_user, corr))
    return self.__convert_mutual_info_list_to_df(neighbour_mutual_info)

  @staticmethod
  def __convert_mutual_info_list_to_df(mutual_info_list):
    mi_df = pd.DataFrame(mutual_info_list)
    if mi_df.empty:
      return pd.DataFrame()
    mi_df.columns = ['user_id', 'correlation']
    mi_df.set_index('user_id', inplace=True)
    return mi_df

  @staticmethod
  def get_n_rating(common_ratings, is_first_user: bool, rating):
    if is_first_user:
      return common_ratings.loc[common_ratings['rating_x'] == rating].count()[0]
    else:
      return common_ratings.loc[common_ratings['rating_y'] == rating].count()[0]

  @staticmethod
  def get_n_rating_pair(common_ratings, user1_rating, user2_rating):
    return common_ratings.loc[(common_ratings['rating_x'] == user1_rating) &
                              (common_ratings['rating_y'] == user2_rating)].count()[0]

  def mutual_information(self, user1_id, user2_id):
    ratings = self.dataset_optimizer.get_ratings()
    common_ratings = pd.merge(ratings.loc[(ratings['user_id'] == user1_id)],
                              ratings.loc[(ratings['user_id'] == user2_id)],
                              on="item_id")
    n_common = common_ratings.count()[0]
    if n_common == 0:
      return 0.0
    # Calculate entropies
    user1_entropy, n1 = self.user_entropy(common_ratings, True, n_common)
    user2_entropy, n2 = self.user_entropy(common_ratings, False, n_common)
    user1_and_2_entropy, n1_2 = self.cross_entropy(common_ratings, n_common)

    # Handle bias of the entropy values and calculate mutual information
    I = user1_entropy + ((n1 - 1) / (2 * n_common))  # User1 entropy
    I += user2_entropy + ((n2 - 1) / (2 * n_common))  # Plus User2 entropy
    I -= user1_and_2_entropy + ((n1_2 - 1) / (2 * n_common))  # Plus Cross entropy
    return I

  def user_entropy(self, common_ratings, is_first_user: bool, n_common):
    entropy = 0.0
    rating = self.lowest_rating
    n = 0
    while rating <= self.highest_rating:
      n_rating = self.get_n_rating(common_ratings, is_first_user, rating)
      Pi = n_rating / n_common
      if Pi > 0:
        entropy -= Pi * math.log(Pi)
        n += 1
      rating += self.rating_increment
    return entropy, n

  def cross_entropy(self, common_ratings, n_common):
    entropy = 0.0
    user1_rating = self.lowest_rating
    n = 0
    while user1_rating <= self.highest_rating:
      user2_rating = self.lowest_rating
      while user2_rating <= self.highest_rating:
        n_rating = self.get_n_rating_pair(common_ratings, user1_rating, user2_rating)
        Pi = n_rating / n_common
        if Pi > 0:
          entropy -= Pi * math.log(Pi)
          n += 1
        user2_rating += self.rating_increment
      user1_rating += self.rating_increment
    return entropy, n
