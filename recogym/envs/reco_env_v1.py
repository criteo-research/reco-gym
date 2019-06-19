# Omega is the users latent representation of interests - vector of size K
#     omega is initialised when you have new user with reset
#     omega is updated at every timestep using timestep
#   
# Gamma is the latent representation of organic products (matrix P by K)
# softmax(Gamma omega) is the next item probabilities (organic)

# Beta is the latent representation of response to actions (matrix P by K)
# sigmoid(beta omega) is the ctr for each action

from numpy import array, diag, exp, matmul, mod, sqrt
from scipy.special import expit as sigmoid
from .abstract import AbstractEnv, env_args, organic

# Default arguments for toy environment ------------------------------------

# inherit most arguments from abstract class
env_1_args = {
    **env_args,
    **{
        'K': 5,
        'sigma_omega_initial': 1,
        'sigma_omega': 0.1,
        'number_of_flips': 0,
        'sigma_mu_organic': 3,
        'change_omega_for_bandits': False,
        'normalize_beta': False
    }
}


# Maps behaviour into ctr - organic has real support ctr is on [0,1].
def ff(xx, aa = 5, bb = 2, cc = 0.3, dd = 2, ee = 6):
    # Magic numbers give a reasonable ctr of around 2%.
    return sigmoid(aa * sigmoid(bb * sigmoid(cc * xx) - dd) - ee)


# Environment definition.
class RecoEnv1(AbstractEnv):

    def __init__(self):
        super(RecoEnv1, self).__init__()

    def set_static_params(self):
        # Initialise the state transition matrix which is 3 by 3
        # high level transitions between organic, bandit and leave.
        self.state_transition = array([
            [0, self.config.prob_organic_to_bandit, self.config.prob_leave_organic],
            [self.config.prob_bandit_to_organic, 0, self.config.prob_leave_organic],
            [0.0, 0.0, 1.]
        ])

        self.state_transition[0, 0] = 1 - sum(self.state_transition[0, :])
        self.state_transition[1, 1] = 1 - sum(self.state_transition[1, :])

        # Initialise Gamma for all products (Organic).
        self.Gamma = self.rng.normal(
            size = (self.config.num_products, self.config.K)
        )

        # Initialise mu_organic.
        self.mu_organic = self.rng.normal(
            0, self.config.sigma_mu_organic,
            size = (self.config.num_products)
        )

        # Initialise beta, mu_bandit for all products (Bandit).
        self.generate_beta(self.config.number_of_flips)

    # Create a new user.
    def reset(self, user_id = 0):
        super().reset(user_id)
        self.omega = self.rng.normal(
            0, self.config.sigma_omega_initial, size = (self.config.K, 1)
        )

    # Update user state to one of (organic, bandit, leave) and their omega (latent factor).
    def update_state(self):
        self.state = self.rng.choice(3, p = self.state_transition[self.state, :])
        assert (hasattr(self, 'time_generator'))
        old_time = self.current_time
        self.current_time = self.time_generator.new_time()
        time_delta = self.current_time - old_time
        omega_k = 1 if time_delta == 0 else time_delta

        # And update omega.
        if self.config.change_omega_for_bandits or self.state == organic:
            self.omega = self.rng.normal(
                self.omega,
                self.config.sigma_omega * omega_k, size = (self.config.K, 1)
            )

    # Sample a click as response to recommendation when user in bandit state
    # click ~ Bernoulli().
    def draw_click(self, recommendation):
        # Personalised CTR for every recommended product.
        ctr = ff(matmul(self.beta, self.omega)[:, 0] + self.mu_bandit)
        click = self.rng.choice(
            [0, 1],
            p = [1 - ctr[recommendation], ctr[recommendation]]
        )
        return click

    # Sample the next organic product view.
    def update_product_view(self):
        log_uprob = matmul(self.Gamma, self.omega)[:, 0] + self.mu_organic
        log_uprob = log_uprob - max(log_uprob)
        uprob = exp(log_uprob)
        self.product_view = int(
            self.rng.choice(
                self.config.num_products,
                p = uprob / sum(uprob)
            )
        )

    def normalize_beta(self):
        self.beta = self.beta / sqrt((self.beta**2).sum(1)[:,None])

    def generate_beta(self, number_of_flips):
        """Create Beta by flipping Gamma, but flips are between similar items only"""

        if number_of_flips == 0:
            self.beta = self.Gamma
            self.mu_bandit = self.mu_organic
            if self.config.normalize_beta:
                self.normalize_beta()

            return
        P, K = self.Gamma.shape
        index = list(range(P))

        prod_cov = matmul(self.Gamma, self.Gamma.T)
        prod_cov = prod_cov - diag(
            diag(prod_cov))  # We are always most correlated with ourselves so remove the diagonal.

        prod_cov_flat = prod_cov.flatten()

        already_used = dict()
        flips = 0
        pcs = prod_cov_flat.argsort()[::-1]  # Find the most correlated entries
        for ii, jj in [(int(p / P), mod(p, P)) for p in pcs]:  # Convert flat indexes to 2d indexes
            # Do flips between the most correlated entries
            # provided neither the row or col were used before.
            if not (ii in already_used or jj in already_used):
                index[ii] = jj  # Do a flip.
                index[jj] = ii
                already_used[ii] = True  # Mark as dirty.
                already_used[jj] = True
                flips += 1

                if flips == number_of_flips:
                    self.beta = self.Gamma[index, :]
                    self.mu_bandit = self.mu_organic[index]
                    if self.config.normalize_beta:
                        self.normalize_beta()
                    return

        self.beta = self.Gamma[index, :]
        self.mu_bandit = self.mu_organic[index]

        if self.config.normalize_beta:
            self.normalize_beta()
