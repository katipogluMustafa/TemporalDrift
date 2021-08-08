from internal.platform.constraints.time_constraint import TimeConstraint
import pandas as pd
from internal.platform.constraints.interval import Interval


class DatasetOperator:

  @staticmethod
  def apply_time_constraint(data: pd.DataFrame, interval: Interval):
    if TimeConstraint.is_valid_max_limit(interval):
      return DatasetOperator.__apply_max_limit_time_constraint(data, interval)

    if TimeConstraint.is_valid_timebin(interval):
      return DatasetOperator.__apply_timebin_time_constraint(data, interval)

    return data

  @staticmethod
  def __apply_max_limit_time_constraint(data: pd.DataFrame, interval: Interval):
    _, end_dt = interval.get_interval()
    return data.loc[data.timestamp < end_dt]

  @staticmethod
  def __apply_timebin_time_constraint(data: pd.DataFrame, interval: Interval):
    start_dt, end_dt = interval.get_interval()
    return data.loc[(data.timestamp >= start_dt) & (data.timestamp < end_dt)]
