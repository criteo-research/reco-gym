from numpy.random.mtrand import RandomState
import numpy as np

from .abstract import Agent

epsilon_greedy_args = {
    'epsilon': 0.01,
    'random_seed': np.random.randint(2 ** 31 - 1),

    # Select an Action that is ABSOLUTELY different to the Action
    # that would have been selected in case when Epsilon-Greedy Policy Selection
    # had not been applied.
    'epsilon_pure_new': True,

    # Try to select the worse case in epsilon-case.
    'epsilon_select_worse': False,
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
            if self.config.epsilon_select_worse:
                product_probas = greedy_action['ps-a']
                product_probas = (1.0 - product_probas)  # Inversion of probabilities.
            else:
                product_probas = np.ones(self.config.num_products)

            if self.config.epsilon_pure_new:
                product_probas[greedy_action['a']] = 0.0
            product_probas = product_probas / np.sum(product_probas)
            epsilon_action = self.rng.choice(
                self.config.num_products,
                p = product_probas
            )
            return {
                **super().act(observation, reward, done),
                **{
                    'a': epsilon_action,
                    'ps': self.config.epsilon * product_probas[epsilon_action],
                    'ps-a': self.config.epsilon * product_probas,
                    'greedy': False,
                    'h0': greedy_action['a']
                }
            }
        else:
            return {
                **greedy_action,
                'greedy': True,
                'ps': (1.0 - self.config.epsilon) * greedy_action['ps'],
                'ps-a': (1.0 - self.config.epsilon) * greedy_action['ps-a'],
            }

    def reset(self):
        self.agent.reset()
