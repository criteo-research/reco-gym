from enum import Enum


class AgentStats(Enum):
    """
    Agent Statistics
    """

    # Confidence Interval.
    Q0_025 = 0
    Q0_500 = 1
    Q0_975 = 2

    # Number of Samples (Users) in a Training Data.
    SAMPLES = 3

    AGENTS = 4
    SUCCESSES = 5  # Amount of Clicks.
    FAILURES = 6  # Amount of non CLicks.


class AgentInit(Enum):
    """
    Abstract data for Agent Initialisation.
    """
    CTOR = 0  # Agent Constructor.
    DEF_ARGS = 1  # Default Agent Arguments.


class TrainingApproach(Enum):
    """
    Training Approach of Evolution of Environment (Explore/Exploit approach).
    """
    ALL_DATA = 0  # All training data should be used (accumulated).
    SLIDING_WINDOW_ALL_DATA = 1  # Fixed amount of training data (sliding window).
    ALL_EXPLORATION_DATA = 2  # All training data obtained during Exploration (accumulated).
    SLIDING_WINDOW_EXPLORATION_DATA = 3  # Fixed amount of training data obtained during Exploration.
    MOST_VALUABLE = 4  # The most valuable training data.
    LAST_STEP = 5  # All data BUT obtained only during the last step (both Explore and Exploit).


class EvolutionCase(Enum):
    """
    Evolution Stats Data.
    """
    SUCCESS = 0
    FAILURE = 1
    ACTIONS = 2
    SUCCESS_GREEDY = 3
    FAILURE_GREEDY = 4


class RoiMetrics(Enum):
    """
    Return of Investment Data.
    """
    ROI_MEAN = 0
    ROI_0_025 = 1
    ROI_0_975 = 2
    ROI_SUCCESS = 3
    ROI_FAILURE = 4
