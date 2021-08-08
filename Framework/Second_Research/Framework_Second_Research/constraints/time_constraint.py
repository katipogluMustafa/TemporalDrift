from internal.platform.constraints.interval import Interval, TimebinInterval, MaxLimitInterval

class TimeConstraint(object):

  @staticmethod
  def __is_expected_class(interval: Interval, target_class: type):
    if interval is None:
      return False

    if not isinstance(interval, target_class):
      return False

    return True

  @staticmethod
  def is_valid_timebin(interval: Interval):
    return TimeConstraint.__is_expected_class(interval, TimebinInterval) and interval.is_valid()

  @staticmethod
  def is_valid_max_limit(interval: Interval):
    return TimeConstraint.__is_expected_class(interval, MaxLimitInterval) and interval.is_valid()
