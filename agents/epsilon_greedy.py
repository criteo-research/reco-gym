from numpy.random.mtrand import RandomState
import numpy as np

from .abstract import Agent

epsilon_greedy_args = {
    'epsilon': 0.1,
}


class EpsilonGreedy(Agent):
    def __init__(self, config, agent):
        super(EpsilonGreedy, self).__init__(config)
        self.agent = agent
        self.rng = RandomState(self.config.random_seed)

    def train(self, observation, action, reward, done = False):
        self.agent.train(observation, action, reward, done)

    def act(self, observation, reward, done):
        greedy_action = self.agent.act(observation, reward, done)

        if self.rng.choice([True, False], p = [self.config.epsilon, 1.0 - self.config.epsilon]):
            all_products = np.arange(self.config.num_products)
            return {
                **super().act(observation, reward, done),
                **{
                    'a': self.rng.choice(
                        all_products[all_products != greedy_action['a']]
                    ),
                    'ps': self.config.epsilon / (self.config.num_products - 1),
                    'greedy': False,
                    'h0': self.agent.act(observation, reward, done)['a']
                }
            }
        else:
            return {
                **greedy_action,
                'greedy': True,
                'ps': (1.0 - self.config.epsilon) * greedy_action['ps']
            }

    def reset(self):
        self.agent.reset()
