import numpy as np


class Agent:
    """
    This is an abstract Agent class.
   The class defines an interface with methods those should be overwritten for a new Agent.
    """

    def __init__(self, env):
        self.env = env

    def act(self, observation, reward, done):
        """An act method takes in an observation, which could either be
           `None` or an Organic_Session (see reco_gym/session.py) and returns
           a integer between 0 and num_products indicating which product the
           agent recommends"""
        if observation is not None:
            # process Organic_Session here
            pass

        # for now just return random random recommendation
        action = np.random.choice(self.env.action_space)
        return {
            'a': action,
            'ps': 1.0 / float(self.env.action_space),
        }

    def train(self, observation, action, reward):
        """Use this function to update your model based on observation, action,
            reward tuples"""
        pass
