import numpy as np
from scipy.special import expit

from recogym import Configuration
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)
from recogym.agents.organic_count import to_categorical

bayesian_poly_args = {
    'num_products': 10,
    'random_seed': np.random.randint(2 ** 31 - 1),

    'poly_degree': 2,
    'max_iter': 5000,
    'aa': 1.,
    'bb': 1.
}

from scipy import rand
from numpy.linalg import inv


# Algorithm 6
# http://www.maths.usyd.edu.au/u/jormerod/JTOpapers/Ormerod10.pdf
def JJ(zeta):
    return 1. / (2. * zeta) * (1. / (1 + np.exp(-zeta)) - 0.5)


# TODO replace explicit inv with linear solves
def bayesian_logistic(Psi, y, mu_beta, Sigma_beta, iter = 200):
    zeta = rand(Psi.shape[0])
    for _ in range(iter):
        q_Sigma = inv(inv(Sigma_beta) + 2 * np.matmul(np.matmul(Psi.T, np.diag(JJ(zeta))), Psi))
        q_mu = np.matmul(q_Sigma, (np.matmul(Psi.T, y - 0.5) + np.matmul(inv(Sigma_beta), mu_beta)))
        zeta = np.sqrt(np.diag(np.matmul(np.matmul(Psi, q_Sigma + np.matmul(q_mu, q_mu.T)), Psi.T)))
    return q_mu, q_Sigma


from scipy.stats import multivariate_normal


class BayesianModelBuilderVB(AbstractFeatureProvider):
    def __init__(self, config):
        super(BayesianModelBuilderVB, self).__init__(config)

    def build(self):
        class BayesianFeaturesProviderVB(ViewsFeaturesProvider):
            """
            """

            def __init__(self, config):
                super(BayesianFeaturesProviderVB, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class BayesianRegressionModelVB(Model):
            """
            """

            def __init__(self, config, Lambda):
                super(BayesianRegressionModelVB, self).__init__(config)
                self.Lambda = Lambda

            def act(self, observation, features):
                X = features
                P = X.shape[1]
                A = np.eye(P)
                XA = np.kron(X, A)

                action_proba = expit(np.matmul(XA, self.Lambda.T)).mean(1)
                action = np.argmax(action_proba)
                ps_all = np.zeros(self.config.num_products)
                ps_all[action] = 1.0
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': 1.0,
                        'ps-a': ps_all,
                    },
                }

        features, actions, deltas, pss = self.train_data()

        X = features
        N = X.shape[0]
        P = X.shape[1]
        A = to_categorical(actions, P)
        XA = np.array([np.kron(X[n, :], A[n, :]) for n in range(N)])
        y = deltas  # clicks

        Sigma = np.kron(self.config.aa * np.eye(P) + self.config.bb,
                        self.config.aa * np.eye(P) + self.config.bb)

        q_mu, q_Sigma = bayesian_logistic(XA, y.reshape((N, 1)),
                                          mu_beta = -6 * np.ones((P ** 2, 1)), Sigma_beta = Sigma)
        Lambda = multivariate_normal.rvs(q_mu.reshape(P ** 2), q_Sigma, 1000)

        # stan version of the above (seems to agree well)
        # fit = pystan.stan('model.stan', data = {'N': features.shape[0], 'P': features.shape[1], 'XA': XA, 'y': y, 'Sigma': Sigma}, chains = 1)
        # s = fit.extract()
        # Lambda = s['lambda']

        ### 

        return (
            BayesianFeaturesProviderVB(self.config),  # Poly is a bad name ..
            BayesianRegressionModelVB(self.config, Lambda)
        )


class BayesianAgentVB(ModelBasedAgent):
    """
    Bayesian Agent.

    Note: the agent utilises VB to train a model.
    """

    def __init__(self, config = Configuration(bayesian_poly_args)):
        print('ffq')
        super(BayesianAgentVB, self).__init__(
            config,
            BayesianModelBuilderVB(config)
        )
