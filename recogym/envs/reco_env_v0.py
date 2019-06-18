from numpy import array, sqrt, kron, eye, ones
from .abstract import AbstractEnv, organic, bandit, stop, f, env_args

# Default arguments for toy environment ------------------------------------

env_0_args = env_args

# Users are grouped into distinct clusters to prevent mixing.
env_args['num_clusters'] = 2

# Variance of the difference between organic and bandit.
env_args['phi_var'] = 0.1


# Environment definition ----------------------------------------------------
class RecoEnv0(AbstractEnv):

    def __init__(self):
        super(RecoEnv0, self).__init__()

    def set_static_params(self):
        # State transition Matrix between Organic, Bandit, Leave
        self.state_transition = array([
            [0, self.config.prob_organic_to_bandit, self.config.prob_leave_organic],
            [self.config.prob_bandit_to_organic, 0, self.config.prob_leave_organic],
            [0.0, 0.0, 1.]
        ])

        self.state_transition[0, 0] = 1 - sum(self.state_transition[0, :])
        self.state_transition[1, 1] = 1 - sum(self.state_transition[1, :])

        # Organic Transition Matrix
        cluster_ratio = int(self.config.num_products / self.config.num_clusters)
        ones_mat = ones((cluster_ratio, cluster_ratio))

        T = kron(eye(self.config.num_clusters), ones_mat)
        T = T / kron(T.sum(1), ones((self.config.num_products, 1))).T
        self.product_transition = T

        # creating click probability matrix
        self.phi = self.rng.normal(
            scale = sqrt(self.config.phi_var),
            size = (self.config.num_products, self.config.num_products)
        )
        self.click_probs = f(self.config.num_products / 5. * (T + T.T) + self.phi)

        self.initial_product_probs = \
            ones((self.config.num_products)) / self.config.num_products

    def reset(self, user_id = 0):
        super().reset(user_id)

        # Current Organic product viewed, choose from initial probabilities
        self.product_view = self.rng.choice(
            self.config.num_products, p = self.initial_product_probs
        )

    def update_state(self):
        """Update Markov state between `organic`, `bandit`, or `stop`"""
        self.state = self.rng.choice(3, p = self.state_transition[self.state, :])

    def draw_click(self, recommendation):
        p = self.click_probs[recommendation, self.product_view]
        return self.rng.binomial(1, p)

    def update_product_view(self):
        probs = self.product_transition[self.product_view, :]
        self.product_view = self.rng.choice(self.config.num_products, p = probs)
