# omega is the users latent representation of interests - vector of size K
#     omega is initialised when you have new user with reset
#     omega is updated at every timestep using timestep
#   
# Gamma is the latent representation of organic products (matrix P by K)
# softmax(Gamma omega) is the next item probabilities (organic)

# beta is the latent representation of response to actions (matrix P by K)
# sigmoid(beta omega) is the ctr for each action

from numpy import array, matmul, exp, diag, mod
from scipy.special import expit as sigmoid
# change name of function since it's confusing

from .abstract import AbstractEnv, organic, bandit, stop, f, env_args

# Default arguements for toy environment ------------------------------------

# inherit most arguments from abstract class
env_1_args = env_args

# RecoEnv1 specific arguments
env_1_args['K'] = 5
env_1_args['sigma_omega_initial'] = 0.01
env_1_args['sigma_omega'] = 1.
env_1_args['number_of_flips'] = 0
env_1_args['sigma_mu_organic'] = 30


# maps behaviour into ctr - organic has real support ctr is on [0,1]
def ff(xx, aa = 5, bb = 2, cc = 0.3, dd = 2, ee = 6):
    return sigmoid(aa * sigmoid(bb * sigmoid(cc * xx) - dd) - ee)  # magic numbers give a reasonable ctr of around 2%


# Environment definition ----------------------------------------------------
class RecoEnv1(AbstractEnv):

    def set_static_params(self):
        # Initialise the state transition matrix which is 3 by 3
        # high level transitions between organic, bandit and leave
        self.state_transition = array([
            [0, self.prob_organic_to_bandit, self.prob_leave_organic],
            [self.prob_bandit_to_organic, 0, self.prob_leave_organic],
            [0.0, 0.0, 1.]
        ])

        self.state_transition[0, 0] = 1 - sum(self.state_transition[0, :])
        self.state_transition[1, 1] = 1 - sum(self.state_transition[1, :])

        # initialise Gamma for all products (organic)
        self.Gamma = self.rng.normal(
            size = (self.num_products, self.K)
        )

        # initialise mu_organic
        self.mu_organic = self.rng.normal(
            0, self.sigma_mu_organic,
            size = (self.num_products)
        )

        # initialise beta, mu_bandit for all products (bandit)
        self.generate_beta(self.number_of_flips)

    # create a new user
    def reset(self):
        super().reset()
        self.omega = self.rng.normal(
            0, self.sigma_omega_initial, size = (self.K, 1)
        )

    # update user state to one of (organic, bandit, leave) and their omega (latent factor)
    def update_state(self):
        self.state = self.rng.choice(3, p = self.state_transition[self.state, :])

        # and update omega
        self.omega = self.rng.normal(self.omega,
                                     self.sigma_omega, size = (self.K, 1)
                                     )

    # sample a click as response to recommendation when user in bandit state
    # click ~ Bernoulli()
    def draw_click(self, recommendation):
        ctr = ff(matmul(self.beta, self.omega)[:, 0] + self.mu_bandit)  # personalised ctr for every recommended product
        click = self.rng.choice(
            [0, 1],
            p = [1 - ctr[recommendation], ctr[recommendation]]
        )
        return click

    # sample the next organic product view
    def update_product_view(self):
        log_uprob = matmul(self.Gamma, self.omega)[:, 0] + self.mu_organic
        log_uprob = log_uprob - max(log_uprob)
        uprob = exp(log_uprob)
        self.product_view = self.rng.choice(
            self.num_products,
            p = uprob / sum(uprob)
        )

    def generate_beta(self, number_of_flips):
        """create Beta by flipping Gamma, but flips are between similar items only"""
        if number_of_flips == 0:
            self.beta = self.Gamma
            self.mu_bandit = self.mu_organic
            return
        P, K = self.Gamma.shape
        index = list(range(P))

        prod_cov = matmul(self.Gamma, self.Gamma.T)
        prod_cov = prod_cov - diag(
            diag(prod_cov))  # we are always most correlated with ourselves so remove the diagonal

        prod_cov_flat = prod_cov.flatten()

        already_used = dict()
        flips = 0
        pcs = prod_cov_flat.argsort()[::-1]  # find the most correlated entries
        for ii, jj in [(int(p / P), mod(p, P)) for p in pcs]:  # convert flat indexes to 2d indexes
            if not (
                    ii in already_used or jj in already_used):  # do flips between the most correlated entries provided neither the row or col were used before
                index[ii] = jj  # do flip
                index[jj] = ii
                already_used[ii] = True  # mark as dirty
                already_used[jj] = True
                flips += 1

                if flips == number_of_flips:
                    self.beta = self.Gamma[index, :]
                    self.mu_bandit = self.mu_organic[index]
                    return

        self.beta = self.Gamma[index, :]
        self.mu_bandit = self.mu_organic[index]


number_of_products = 10


def BuildLogisticRegressionData(
        data,
        number_of_users,
        only_with_clicks = False,
        sliding_window_depth = -1
):
    train_X = []
    train_Y = []

    for user_id in range(number_of_users):
        views = np.zeros((0, number_of_products))
        for _, user_datum in data[data['u'] == user_id].iterrows():
            if user_datum['c'] == -1:
                view = int(user_datum['v'])
                tmp_view = np.zeros(number_of_products)

                tmp_view[view] = 1
                views = np.append(tmp_view[np.newaxis, :], views, axis = 0)
            else:
                assert (user_datum['c'] != -1)
                assert (user_datum['r'] != -1)
                view = int(user_datum['r'])

                if views.shape[0] <= sliding_window_depth or sliding_window_depth == -1:
                    train_views = views
                else:
                    train_views = views[:sliding_window_depth, :]
                if only_with_clicks:
                    if user_datum['c'] != 0:
                        train_X.append(train_views.sum(axis = 0))
                        train_Y.append(view)
                else:
                    train_X.append(np.append(user_datum['r'], train_views.sum(axis = 0)))
                    train_Y.append(user_datum['c'])

    train_X = np.array(train_X).reshape(
        len(train_Y),
        number_of_products if only_with_clicks else number_of_products + 1
    )
    train_Y = np.array(train_Y)

    return train_X, train_Y


# define agent class
class ModelBasedBackLogistic:
    def __init__(self, env, model):
        # set environment as an attribute of agent
        self.env = env
        self.model = model
        self.organic_views = np.zeros((1, self.env['num_products']))

    def train(self, observations, action, reward, done):
        for observation in observations:
            self.organic_views[:, observation[1]] += 1

    def offline_train(train):
        pass

    def act(self, _ = None):
        '''act method returns an action based on current observation and past
            history'''
        max_prob = -1.0
        for view in range(number_of_products):
            case = np.append(np.array(view).reshape(1, 1), self.organic_views, axis = 1)
            prob = self.model.predict_proba(case)[0, 1]
            if prob > max_prob:
                max_prob = prob
                action = view
        return {
            'a': action,
            'ps': prob
        }

    def reset(self):
        self.organic_views = np.zeros((1, self.env['num_products']))
