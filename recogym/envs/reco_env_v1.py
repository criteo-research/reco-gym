# Omega is the users latent representation of interests - vector of size K
#     omega is initialised when you have new user with reset
#     omega is updated at every timestep using timestep
#
# Gamma is the latent representation of organic products (matrix P by K)
# softmax(Gamma omega) is the next item probabilities (organic)

# Beta is the latent representation of response to actions (matrix P by K)
# sigmoid(beta omega) is the ctr for each action
import numpy as np
from numba import njit

from .abstract import AbstractEnv, env_args, organic
from numpy.random.mtrand import RandomState

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


@njit(nogil=True)
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# Maps behaviour into ctr - organic has real support ctr is on [0,1].
@njit(nogil=True)
def ff(xx, aa=5, bb=2, cc=0.3, dd=2, ee=6):
    # Magic numbers give a reasonable ctr of around 2%.
    return sig(aa * sig(bb * sig(cc * xx) - dd) - ee)


# Environment definition.
class RecoEnv1(AbstractEnv):

    def __init__(self):
        super(RecoEnv1, self).__init__()
        self.cached_state_seed = None

    def set_static_params(self):
        # Initialise the state transition matrix which is 3 by 3
        # high level transitions between organic, bandit and leave.
        self.state_transition = np.array([
            [0, self.config.prob_organic_to_bandit, self.config.prob_leave_organic],
            [self.config.prob_bandit_to_organic, 0, self.config.prob_leave_bandit],
            [0.0, 0.0, 1.]
        ])

        self.state_transition[0, 0] = 1 - sum(self.state_transition[0, :])
        self.state_transition[1, 1] = 1 - sum(self.state_transition[1, :])

        # Initialise Gamma for all products (Organic). mean, var = 0, 1
        self.Gamma = self.rng.normal(
            size=(self.config.num_products, self.config.K)
        )

        # Initialise mu_organic. mean, var = 0, sigma_mu_organic(3)
        self.mu_organic = self.rng.normal(
            0, self.config.sigma_mu_organic,
            size=(self.config.num_products, 1)
        )

        # Initialise beta, mu_bandit for all products (Bandit).
        self.generate_beta(self.config.number_of_flips)

    # Create a new user.
    def reset(self, user_id=0):
        super().reset(user_id)
        if self.config.random_seed_for_user is not None:
            self.omega = self.user_rng.normal(
                0, self.config.sigma_omega_initial, size=(self.config.K, 1)
            ) # mean, var = 0, sigma_omega_initial(1)
        else:
            self.omega = self.rng.normal(
                0, self.config.sigma_omega_initial, size=(self.config.K, 1)
            ) # mean, var = 0, sigma_omega_initial(1)


    # Update user state to one of (organic, bandit, leave) and their omega (latent factor).
    def update_state(self):
        old_state = self.state
        self.state = self.rng.choice(3, p=self.state_transition[self.state, :])
        assert (hasattr(self, 'time_generator'))
        old_time = self.current_time
        self.current_time = self.time_generator.new_time()
        time_delta = self.current_time - old_time
        omega_k = 1 if time_delta == 0 else time_delta

        # And update omega.
        if self.config.change_omega_for_bandits or self.state == organic:
            self.omega = self.rng.normal(
                self.omega,
                self.config.sigma_omega * omega_k, size=(self.config.K, 1)
            ) # mean, var = omega, sigma_omega(1)
        self.context_switch = old_state != self.state

    # Sample a click as response to recommendation when user in bandit state
    # click ~ Bernoulli().
    def draw_click(self, recommendation):
        # Personalised CTR for every recommended product.
        if self.config.change_omega_for_bandits or self.context_switch:
            self.cached_state_seed = (
                    self.beta @ self.omega + self.mu_bandit
            ).ravel()
        assert self.cached_state_seed is not None
        ctr = ff(self.cached_state_seed)
        click = self.rng.choice(
            [0, 1],
            p=[1 - ctr[recommendation], ctr[recommendation]]
        )
        return click, ctr[recommendation]

    # Sample the next organic product view.
    def update_product_view(self):
        log_uprob = (self.Gamma @ self.omega + self.mu_organic).ravel()
        log_uprob = log_uprob - max(log_uprob)
        uprob = np.exp(log_uprob)
        self.product_view = np.int16(
            self.rng.choice(
                self.config.num_products,
                p=uprob / uprob.sum()
            )
        )

    def normalize_beta(self):
        self.beta = self.beta / np.sqrt((self.beta ** 2).sum(1)[:, np.newaxis])

    def generate_beta(self, number_of_flips):
        """Create Beta by flipping Gamma, but flips are between similar items only"""

        if number_of_flips == 0:
            self.beta = self.Gamma
            self.mu_bandit = self.mu_organic
            if self.config.normalize_beta:
                self.normalize_beta()

            return

        P, K = self.Gamma.shape
        index = np.arange(P)

        prod_cov = self.Gamma @ self.Gamma.T
        # We are always most correlated with ourselves so remove the diagonal.
        prod_cov = prod_cov - np.diag(np.diag(prod_cov))

        prod_cov_flat = prod_cov.flatten()

        already_used = set()
        flips = 0
        for p in prod_cov_flat.argsort()[::-1]:  # Find the most correlated entries
            # Convert flat indexes to 2d indexes
            ii, jj = int(p / P), np.mod(p, P)
            # Do flips between the most correlated entries
            # provided neither the row or col were used before.
            if not (ii in already_used or jj in already_used):
                index[ii] = jj  # Do a flip.
                index[jj] = ii
                already_used.add(ii)
                already_used.add(jj)
                flips += 1

                if flips == number_of_flips:
                    break

        self.beta = self.Gamma[index, :]
        self.mu_bandit = self.mu_organic[index, :]

        if self.config.normalize_beta:
            self.normalize_beta()
