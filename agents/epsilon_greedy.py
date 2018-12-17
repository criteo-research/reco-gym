from numpy.random.mtrand import RandomState

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
        if self.rng.choice([True, False], p = [self.config.epsilon, 1.0 - self.config.epsilon]):
            return {
                **super().act(observation, reward, done),
                **{
                    'a': self.rng.choice(self.config.num_products),
                    'ps': 1.0 / self.config.num_products,
                    'e-g': True,
                    'h0': self.agent.act(observation, reward, done)['a']
                }
            }
        else:
            return {
                **self.agent.act(observation, reward, done),
                'e-g': False,
            }
