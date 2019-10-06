import numpy as np

from recogym import Configuration, build_agent_init, to_categorical
from recogym.agents import Agent

test_agent_args = {
    'num_products': 10,
    'with_ps_all': False,
}


class TestAgent(Agent):
    """Organic counter agent"""

    def __init__(self, config = Configuration(test_agent_args)):
        super(TestAgent, self).__init__(config)

        self.co_counts = np.zeros((self.config.num_products, self.config.num_products))
        self.corr = None

    def act(self, observation, reward, done):
        """Make a recommendation"""

        self.update_lpv(observation)

        action = self.co_counts[self.last_product_viewed, :].argmax()
        if self.config.with_ps_all:
            ps_all = np.zeros(self.config.num_products)
            ps_all[action] = 1.0
        else:
            ps_all = ()
        return {
            **super().act(observation, reward, done),
            **{
                'a': self.co_counts[self.last_product_viewed, :].argmax(),
                'ps': 1.0,
                'ps-a': ps_all,
            },
        }

    def train(self, observation, action, reward, done = False):
        """Train the model in an online fashion"""
        if observation.sessions():
            A = to_categorical(
                [session['v'] for session in observation.sessions()],
                self.config.num_products
            )
            B = A.sum(0).reshape((self.config.num_products, 1))
            self.co_counts = self.co_counts + np.matmul(B, B.T)

    def update_lpv(self, observation):
        """updates the last product viewed based on the observation"""
        if observation.sessions():
            self.last_product_viewed = observation.sessions()[-1]['v']


