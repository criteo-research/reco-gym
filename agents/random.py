import numpy as np
from numpy.random import choice

# Default Arguments ----------------------------------------------------------
random_args = {}

random_args['num_products'] = 10


class RandomAgent(object):

    """The world's simplest agent!"""
    def __init__(self, args):
        # set all key word arguments as attributes
        for key in args:
            setattr(self, key, args[key])

    def act(self, observation, reward, done):
        return choice(self.num_products)

    def train(self, observation, action, reward, done):
        pass
