from internal.platform.trecs.dataset import Dataset
from datetime import datetime,date
from collections import defaultdict

class UserRating:
    def __init__(self, dataset: Dataset):
        dataset.load_ratings()
        self.ratings = dataset.ratings

    def user_ratings(self, uid):
        if self.ratings is None:
            self.ratings = Dataset.get_ratings_from_file()

        user_ratings = []
        for user_id, item_id, rating, timestamp in self.ratings:
            if user_id == uid:
                user_ratings.append((item_id, rating, timestamp))

        return user_ratings

    def user_avg_ratings(self, uid):
        ratings_sum = 0
        count = 0

        for _, rating, _ in self.user_ratings(uid):
            ratings_sum += rating
            count += 1

        return ratings_sum / count

    def user_avg_timestamp(self, uid):
        timestamp_sum = 0
        count = 0

        for _, _, timestamp in self.user_ratings(uid):
            timestamp_sum += timestamp
            count += 1

        return timestamp_sum / count

    def user_avg_rating_date(self, uid) -> date:
        return datetime.fromtimestamp(self.user_avg_timestamp(uid)).date()

    def user_active_day_count(self, uid):
        active_days = defaultdict(int)
        for _, _, timestamp in self.user_ratings(uid):
            active_days[datetime.fromtimestamp(timestamp).date()] += 1

        return len(active_days)

    def user_active_dates(self, uid) -> defaultdict:
        """
        Get defaultdict of uniq days with number of rating activity in that day
        :param uid: User id
        :return: defaultdict{date->n_of_activity}
        """
        active_days = defaultdict(int)
        for _, _, timestamp in self.user_ratings(uid):
            active_days[datetime.fromtimestamp(timestamp).date()] += 1

        return active_days

    ## User Activity Per Day of Year

    def user_activity_per_day_of_year(self, uid) -> defaultdict:
        days_of_year = defaultdict(int)
        for current_activity_date in self.user_active_dates(uid):
            days_of_year[current_activity_date.timetuple().tm_yday] += 1

        return days_of_year

    def user_activity_per_day_of_year_list(self, uid):
        return UserRating._user_activity_get_list(self.user_activity_per_day_of_year, uid)

    def user_activity_per_day_of_year_list_sorted(self, uid):
        return UserRating._user_activity_get_list_sorted(self.user_activity_per_day_of_year_list, uid)

    ## User Activity Per Week of Year

    def user_activity_per_week(self, uid):
        week_of_year = defaultdict(int)
        for current_activity_date in self.user_active_dates(uid):
            week_of_year[current_activity_date.isocalendar()[1]] += 1

        return week_of_year

    def user_activity_week_list(self, uid):
        return UserRating._user_activity_get_list(self.user_activity_per_week, uid)

    def user_activity_per_week_list_sorted(self, uid):
        return UserRating._user_activity_get_list_sorted(self.user_activity_week_list, uid)

    @staticmethod
    def _user_activity_get_list(data_getter_func, uid):
        activity = []
        for item, count in data_getter_func(uid).items():
            activity.append((item, count))
        return activity

    @staticmethod
    def _user_activity_get_list_sorted(data_getter_func, uid):
        activity = data_getter_func(uid)
        activity.sort(key=lambda tup: tup[1], reverse=True)
        return activity


#ur = UserRating(Dataset())
#print(ur.user_active_day_count(448))
#print(ur.user_active_dates(448))
#print(ur.user_activity_per_day_of_year_tuples(448).sort(reverse=True))
#print(ur.user_activity_per_day_of_year_list_sorted(448))
#print(ur.user_activity_per_week(448))
#print(ur.user_activity_per_week_list_sorted(448))