class TimeConstraint:

    def __init__(self, end_dt, start_dt=None):
        """
        When end_dt is only given, system will have a max time constraint only.

        When end_dt and start_dt are given, system will have beginning end ending boundary.

        :param end_dt: The maximum limit of the time constraint.
        :param start_dt: The minimum limit of the time constraint.
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
        Check whether this TimeConstraint represents a valid max time limit.
        """
        if (self._end_dt is not None) and (self._start_dt is None):
            return True

    def is_time_bin(self) -> bool:
        if (self._start_dt is not None) and (self._end_dt is not None):
            return True
        return False

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


