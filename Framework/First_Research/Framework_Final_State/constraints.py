class TimeConstraint:

    def __init__(self, end_dt, start_dt=None):
        """
        When end_dt is only given, TimeConstraint will be of type max limit.

        When start_dt and end_dt are given, TimeConstraint will be of type time bin.

        :param end_dt: The end time.
        :param start_dt: The start time.
            Always set start_dt to None if you change the object from time_bin to max_limit.
        """
        self.end_dt = end_dt
        self.start_dt = start_dt

    def is_valid_time_bin(self) -> bool:
        """
        Check whether this TimeConstraint object represents a valid time bin.
        """
        if self.is_time_bin() and (self._end_dt > self._start_dt):
            return True
        return False

    def is_valid_max_limit(self) -> bool:
        """
        Check whether the object is of max time limit type.
        """
        if (self._end_dt is not None) and (self._start_dt is None):
            return True

    def is_time_bin(self) -> bool:
        """
        Check whether the object is of time bin type. 
        """
        if (self._start_dt is not None) and (self._end_dt is not None):
            return True
        return False

    # Comparing TimeConstraints

    def __eq__(self, other):
        if other is None:
            return False
        return self._start_dt == other.start_dt and self._end_dt == other.end_dt

    def __ne__(self, other):
        if other is None:
            return False
        return self._start_dt != other.start_dt or self._end_dt != other.end_dt

    # Properties

    @property
    def end_dt(self):
        return self._end_dt

    @end_dt.setter
    def end_dt(self, value):
        self._end_dt = value

    @property
    def start_dt(self):
        return self._start_dt

    @start_dt.setter
    def start_dt(self, value):
        self._start_dt = value

    # Printing TimeConstraints

    def __repr__(self):
        return f"(start = {self._start_dt}, end= {self._end_dt})"

    def __str__(self):
        return f"(start = {self._start_dt}, end= {self._end_dt})"