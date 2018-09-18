import numpy as np

bandit_count_args = {}

bandit_count_args['num_products'] = 10


class BanditCount:
    def __init__(self, args):
        # set all key word arguments as attributes
        for key in args:
            setattr(self, key, args[key])

        self.pulls_a = np.zeros((self.num_products, self.num_products))
        self.clicks_a = np.zeros((self.num_products, self.num_products))
        self.last_product_viewed = None
        self.ctr = (self.clicks_a + 1) / (self.pulls_a + 2)

    def act(self, observation, reward, done):
        """Make a recommendation"""

        self.update_lpv(observation)

        return self.ctr[self.last_product_viewed, :].argmax()

    def train(self, observation, action, reward, done):
        """Train the model in an online fashion"""

        if action is not None and reward is not None:

            self.update_lpv(observation)
            self.pulls_a[self.last_product_viewed, action] += 1
            self.clicks_a[self.last_product_viewed, action] += reward

            self.ctr = (self.clicks_a + 1) / (self.pulls_a + 2)

    def update_lpv(self, observation):
        """updates the last product viewed based on the observation"""
        if observation is not None:
            self.last_product_viewed = observation[-1][-1]

    def save(self, location):
        """Save the state of the model to disk"""

        np.save(location + "pulls_a.npy", self.pulls_a)
        np.save(location + "clicks_a.npy", self.clicks_a)

    def load(self, location):
        """Load the model state from disk"""

        self.pulls_a = np.load(location + "pulls_a.npy")
        self.clicks_a = np.load(location + "clicks_a.npy")
