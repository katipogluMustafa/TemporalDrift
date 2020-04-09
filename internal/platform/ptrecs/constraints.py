from enum import auto, Enum


class PredictionTimeConstraint(Enum):
    """
    PredictionTimeConstraint is used whenever we need to differentiate between time constraints.

    AT -> Set the RecSys at a specific system time, so that anything after given time wont be taken into account.
    IN -> Set the RecSys in between two specific time intervals
    NO -> Set no boundaries for RecSys, All ratings and neighbours will be used.
    """
    AT = auto()
    IN = auto()
    NO = auto()