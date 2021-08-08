import unittest
from datetime import datetime

from internal.platform.constraints.interval import TimebinInterval, MaxLimitInterval, Interval
from internal.platform.constraints.time_constraint import TimeConstraint


class TestTimeConstraint(unittest.TestCase):

  def test_timebin_constraint(self):
    self.assert_timebin_constraint(None, False)
    self.assert_timebin_constraint(TimebinInterval(None, datetime(2006, 5, 5)), False)
    self.assert_timebin_constraint(TimebinInterval(datetime(2006, 5, 5), None), False)
    self.assert_timebin_constraint(TimebinInterval(None, None), False)
    self.assert_timebin_constraint(TimebinInterval(datetime(2005, 5, 5), datetime(2006, 5, 5)), True)

  def assert_timebin_constraint(self, time_interval: Interval, assertion: bool):
    return self.assertEqual(TimeConstraint.is_valid_timebin(time_interval), assertion)

  def test_max_limit_constraint(self):
    self.assert_max_limit_constraint(None, False)
    self.assert_max_limit_constraint(MaxLimitInterval(datetime(2006, 5, 5), None), False)
    self.assert_max_limit_constraint(MaxLimitInterval(None, None), False)
    self.assert_max_limit_constraint(MaxLimitInterval(datetime(2005, 5, 5), datetime(2006, 5, 5)), False)
    self.assert_max_limit_constraint(MaxLimitInterval(None, datetime(2006, 5, 5)), True)

  def assert_max_limit_constraint(self, time_interval: Interval, assertion: bool):
    return self.assertEqual(TimeConstraint.is_valid_max_limit(time_interval), assertion)

if __name__ == '__main__':
  unittest.main(argv=[''], verbosity=2, exit=False)