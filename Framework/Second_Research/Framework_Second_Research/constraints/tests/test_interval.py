import unittest
from datetime import datetime
from internal.platform.constraints.interval import Interval, TimebinInterval, MaxLimitInterval


class TestInterval(unittest.TestCase):

  def test_left_open_interval(self):
    self.assert_left_open(Interval(None, datetime(2020, 5, 5)), True)
    self.assert_left_open(Interval(None, None), False)
    self.assert_left_open(Interval(datetime(2020, 5, 5), None), False)
    self.assert_left_open(Interval(datetime(2020, 5, 5), datetime(2020, 5, 5)), False)

  def test_closed_interval(self):
    self.assert_closed(Interval(None, None), False)
    self.assert_closed(Interval(None, datetime(2020, 5, 5)), False)
    self.assert_closed(Interval(datetime(2020, 5, 5), None), False)
    self.assert_closed(Interval(datetime(2000, 5, 5), datetime(2020, 5, 5)), True)
    self.assert_closed(Interval(datetime(2020, 5, 5), datetime(2000, 5, 5)), False)

  def assert_left_open(self, time_constraint: Interval, assertion: bool):
    return self.assertEqual(time_constraint.is_left_open_time_interval(), assertion)

  def assert_closed(self, time_constraint: Interval, assertion: bool):
    return self.assertEqual(time_constraint.is_closed_time_interval(), assertion)


class TestTimebinInterval(unittest.TestCase):

  def test_valid_timebin(self):
    self.assert_timebin(TimebinInterval(), False)
    self.assert_timebin(TimebinInterval(None), False)
    self.assert_timebin(TimebinInterval(None, None), False)
    self.assert_timebin(TimebinInterval(datetime(2000, 5, 5), None), False)
    self.assert_timebin(TimebinInterval(None, datetime(2000, 5, 5)), False)
    self.assert_timebin(TimebinInterval(datetime(2000, 5, 5), datetime(2020, 5, 5)), True)

  def assert_timebin(self, timebin_constraint: TimebinInterval, assertion: bool):
    return self.assertEqual(timebin_constraint.is_valid(), assertion)


class TestMaxLimitInterval(unittest.TestCase):
  def test_valid_max_limit(self):
    self.assert_max_limit(MaxLimitInterval(None), False)
    self.assert_max_limit(MaxLimitInterval(None, None), False)
    self.assert_max_limit(MaxLimitInterval(datetime(2000, 5, 5), None), False)
    self.assert_max_limit(MaxLimitInterval(datetime(2000, 5, 5), datetime(2001, 5, 5)), False)
    self.assert_max_limit(MaxLimitInterval(None, datetime(2001, 5, 5)), True)

  def assert_max_limit(self, max_limit_constraint: MaxLimitInterval, assertion: bool):
    return self.assertEqual(max_limit_constraint.is_valid(), assertion)


if __name__ == '__main__':
  unittest.main(argv=[''], verbosity=2, exit=False)
