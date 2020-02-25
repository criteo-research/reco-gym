import numpy as np

from ..envs.configuration import Configuration

from .abstract import Agent

bandit_count_args = {
    'num_products': 10,
    'with_ps_all': False,
}


class BanditCount(Agent):
    """
    Bandit Count

    The Agent that selects an Action for the most frequently clicked Action before.
    """

    def __init__(self, config = Configuration(bandit_count_args)):
        super(BanditCount, self).__init__(config)

        self.pulls_a = np.zeros((self.config.num_products, self.config.num_products))
        self.clicks_a = np.zeros((self.config.num_products, self.config.num_products))
        self.last_product_viewed = None
        self.ctr = (self.clicks_a + 1) / (self.pulls_a + 2)

    def act(self, observation, reward, done):
        """Make a recommendation"""

        self.update_lpv(observation)
        action = self.ctr[self.last_product_viewed, :].argmax()

        if self.config.with_ps_all:
            ps_all = np.zeros(self.config.num_products)
            ps_all[action] = 1.0
        else:
            ps_all = ()

        return {
            **super().act(observation, reward, done),
            **{
                'a': self.ctr[self.last_product_viewed, :].argmax(),
                'ps': self.ctr[self.last_product_viewed, :][action],
                'ps-a': ps_all,
            },
        }

    def train(self, observation, action, reward, done = False):
        """Train the model in an online fashion"""

        if action is not None and reward is not None:

            ix = self.last_product_viewed
            jx = action['a']
            self.update_lpv(observation)
            self.pulls_a[ix, jx] += 1
            self.clicks_a[ix, jx] += reward

            self.ctr[ix, jx] = (
                    (self.clicks_a[ix, jx] + 1) / (self.pulls_a[ix, jx] + 2)
            )

    def update_lpv(self, observation):
        """Updates the last product viewed based on the observation"""
        if observation.sessions():
            self.last_product_viewed = observation.sessions()[-1]['v']

    def save(self, location):
        """Save the state of the model to disk"""

        np.save(location + "pulls_a.npy", self.pulls_a)
        np.save(location + "clicks_a.npy", self.clicks_a)

    def load(self, location):
        """Load the model state from disk"""

        self.pulls_a = np.load(location + "pulls_a.npy")
        self.clicks_a = np.load(location + "clicks_a.npy")

    def reset(self):
        pass
