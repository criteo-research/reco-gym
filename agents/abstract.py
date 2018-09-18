from numpy.random import choice


class Agent:
    """This is an abstract Agent class. It shows you the methods you need to
        implement to create your own reccomendation agent"""

    def __init__(self, env):
        self.env = env

    def act(self, observation):
        """An act method takes in an observation, which could either be
           `None` or an Organic_Session (see reco_gym/session.py) and returns
           a integer between 0 and num_products indicating which product the
           agent reccomends"""
        if observation is not None:
            # process Organic_Session here
            pass

        # for now just return random random reccomendation
        action = choice(self.env.action_space)
        return action

    def train(self, observation, action, reward):
        """Use this function to update your model based on observation, action,
            reward tuples"""
        pass
