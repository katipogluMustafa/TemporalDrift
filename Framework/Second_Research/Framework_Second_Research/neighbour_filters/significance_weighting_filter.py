class SignificanceWeightingFilter:
  @staticmethod
  def filter(neighbours, correlation_column_name='correlation', n_common_column_name='n_common'):
    neighbours[correlation_column_name] = neighbours[n_common_column_name] * neighbours[correlation_column_name]