import pandas as pd


class MinCorrelationFilter:
  @staticmethod
  def filter(neighbours: pd.DataFrame, minimum_correlation=0.0, correlation_column_name='correlation') -> pd.DataFrame:
    if neighbours is None or neighbours.empty:
      return pd.DataFrame()
    return neighbours[
      MinCorrelationFilter.__has_correlation_more_than_minimum(correlation_column_name, minimum_correlation,
                                                               neighbours)]

  @staticmethod
  def __has_correlation_more_than_minimum(correlation_column_name, minimum_correlation, neighbours):
    return neighbours[correlation_column_name] > minimum_correlation
