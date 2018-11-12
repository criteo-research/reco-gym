# Default Arguments.
from numpy.random.mtrand import choice

random_args = {
    'num_products': 10
}


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, args):
        # Set all key word arguments as attributes.
        for key in args:
            setattr(self, key, args[key])

    def act(self, observation, reward, done):
        return {
            'a': choice(self.num_products),
            'ps': 1.0 / float(self.num_products),
        }

    def train(self, observation, action, reward, done):
        pass
