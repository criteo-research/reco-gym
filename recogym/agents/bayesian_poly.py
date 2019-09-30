import numpy as np
import pystan
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


class BayesianModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(BayesianModelBuilder, self).__init__(config)

    def build(self):
        class BayesianFeaturesProvider(ViewsFeaturesProvider):
            """
            """

            def __init__(self, config):
                super(BayesianFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class BayesianRegressionModel(Model):
            """
            """

            def __init__(self, config, Lambda):
                super(BayesianRegressionModel, self).__init__(config)
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
        fit = pystan.stan('recogym/agents/model.stan', data = {
            'N': features.shape[0],
            'P': features.shape[1],
            'XA': XA,
            'y': deltas,
            'Sigma': Sigma
        }, chains = 1)
        s = fit.extract()
        Lambda = s['lambda']

        return (
            BayesianFeaturesProvider(self.config),  # Poly is a bad name ..
            BayesianRegressionModel(self.config, Lambda)
        )


class BayesianAgent(ModelBasedAgent):
    """
    Bayesian Agent.

    Note: the agent utilises Stan to train a model.
    """

    def __init__(self, config = Configuration(bayesian_poly_args)):
        super(BayesianAgent, self).__init__(
            config,
            BayesianModelBuilder(config)
        )
