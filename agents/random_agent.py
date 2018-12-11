# Default Arguments.
from numpy.random.mtrand import RandomState

from agents import Agent
from reco_gym import Configuration

random_args = {
    'num_products': 10,
    'random_seed': 42,
}


class RandomAgent(Agent):
    """The world's simplest agent!"""

    def __init__(self, config = Configuration(random_args)):
        super(RandomAgent, self).__init__(config)
        self.rng = RandomState(config.random_seed)

    def act(self, observation, reward, done):
        return {
            **super().act(observation, reward, done),
            **{
                'a': self.rng.choice(self.config.num_products),
                'ps': 1.0 / float(self.config.num_products),
            },
        }
