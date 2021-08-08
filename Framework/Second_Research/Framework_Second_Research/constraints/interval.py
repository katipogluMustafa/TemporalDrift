class Interval:

  def __init__(self, interval_beginning_datetime=None, interval_end_datetime=None):
    self.__interval_beginning = interval_beginning_datetime
    self.__interval_end = interval_end_datetime

  def get_interval(self):
    return self.__interval_beginning, self.__interval_end

  def is_valid(self):
    return True

  def is_left_open_time_interval(self):
    if self.__interval_beginning is None and self.__interval_end is not None:
      return True
    return False

  def is_closed_time_interval(self):
    if self.__is_any_end_none():
      return False

    if self.__is_ending_greater_than_beginning():
      return False

    return True

  def get_interval_length_in_days(self):
    return abs((self.__interval_beginning - self.__interval_end).days)


  def __is_any_end_none(self):
    return self.__interval_beginning is None or self.__interval_end is None

  def __is_ending_greater_than_beginning(self):
    return self.__interval_beginning > self.__interval_end

class TimebinInterval(Interval):

  def is_valid(self):
    return self.is_closed_time_interval()

class MaxLimitInterval(Interval):

  def is_valid(self):
    return self.is_left_open_time_interval()
