from abc import ABC

import gym
import numpy as np
import pandas as pd
from gym.spaces import Discrete
from numpy.random.mtrand import RandomState
from scipy.special import expit as sigmoid
from tqdm import trange

from .configuration import Configuration
from .context import DefaultContext
from .features.time import DefaultTimeGenerator
from .observation import Observation
from .session import OrganicSessions
from ..agents import Agent

# Arguments shared between all environments.

env_args = {
    'num_products': 10,
    'num_users': 100,
    'random_seed': np.random.randint(2 ** 31 - 1),
    # Markov State Transition Probabilities.
    'prob_leave_bandit': 0.01,
    'prob_leave_organic': 0.01,
    'prob_bandit_to_organic': 0.05,
    'prob_organic_to_bandit': 0.25,
    'normalize_beta': False,
    'with_ps_all': False
}


# Static function for squashing values between 0 and 1.
def f(mat, offset=5):
    """Monotonic increasing function as described in toy.pdf."""
    return sigmoid(mat - offset)


# Magic numbers for Markov states.
organic = 0
bandit = 1
stop = 2


class AbstractEnvSale(gym.Env, ABC):

    def __init__(self):
        gym.Env.__init__(self)
        ABC.__init__(self)

        self.first_step = True
        self.config = None
        self.state = None
        self.current_user_id = None
        self.current_time = None
        self.empty_sessions = OrganicSessions()

    def reset_random_seed(self, epoch=0):
        # Initialize Random State.
        assert (self.config.random_seed is not None)
        self.rng = RandomState(self.config.random_seed + epoch)

    def init_gym(self, args):

        self.config = Configuration(args)

        # Defining Action Space.
        self.action_space = Discrete(self.config.num_products)

        if 'time_generator' not in args:
            self.time_generator = DefaultTimeGenerator(self.config)
        else:
            self.time_generator = self.config.time_generator

        # Setting random seed for the first time.
        self.reset_random_seed()

        if 'agent' not in args:
            self.agent = None
        else:
            self.agent = self.config.agent

        # Setting any static parameters such as transition probabilities.
        self.set_static_params()

        # Set random seed for second time, ensures multiple epochs possible.
        self.reset_random_seed()

    def reset(self, user_id=0):
        # Current state.
        self.first_step = True
        self.state = organic  # Manually set first state as Organic.

        self.time_generator.reset()
        if self.agent:
            self.agent.reset()

        self.current_time = self.time_generator.new_time()
        self.current_user_id = user_id

        # Record number of times each product seen for static policy calculation.
        self.organic_views = np.zeros(self.config.num_products)


    
   
    
    
    
    
    