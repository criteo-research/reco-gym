import numpy as np

from numpy.random.mtrand import RandomState
from sklearn.linear_model import LogisticRegression

from recogym.agents import *
from recogym import Configuration

logreg_multiclass_ips_args = {
    'num_products': 10,
    'number_of_flips': 1,
    'random_seed': np.random.randint(2 ** 31 - 1),

    # Select a Product randomly with the the probability predicted by Multi-Class Logistic Regression.
    'select_randomly': False,

    'poly_degree': 2,
    'solver': 'lbfgs',
    'max_iter': 5000,
}


class LogregMulticlassIpsModelBuilder(AbstractFeatureProvider):
    """
    Logistic Regression Multiclass Model Builder

    The class that provides both:
    * Logistic Regression Model
    * Feature Provider that builds a Feature Set suitable for the Logistic Regression Model
    """

    def __init__(self, config):
        super(LogregMulticlassIpsModelBuilder, self).__init__(config)
        if config.select_randomly:
            self.rng = RandomState(self.config.random_seed)

    def build(self):
        class LogregMulticlassViewsFeaturesProvider(ViewsFeaturesProvider):
            """
            Logistic Regression Multiclass Feature Provider
            """

            def __init__(self, config):
                super(LogregMulticlassViewsFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class LogregMulticlassModel(Model):
            """
            Logistic Regression Multiclass Model
            """

            def __init__(self, config, logreg):
                super(LogregMulticlassModel, self).__init__(config)
                self.logreg = logreg
                if config.select_randomly:
                    self.rng = RandomState(self.config.random_seed)

            def act(self, observation, features):
                if self.config.select_randomly:
                    action_proba = self.logreg.predict_proba(features)[0, :]
                    action = self.rng.choice(
                        self.config.num_products,
                        p = action_proba
                    )
                    ps = action_proba[action]
                    all_ps = action_proba
                else:
                    action = self.logreg.predict(features).item()
                    ps = 1.0
                    all_ps = np.zeros(self.config.num_products)
                    all_ps[action] = 1.0
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': ps,
                        'ps-a': all_ps,
                    },
                }

        features, actions, deltas, pss = self.train_data()
        weights = deltas / pss

        logreg = LogisticRegression(
            solver = self.config.solver,
            max_iter = self.config.max_iter,
            multi_class = 'multinomial',
            random_state = self.config.random_seed
        )

        lr = logreg.fit(features, actions, weights)

        return (
            LogregMulticlassViewsFeaturesProvider(self.config),
            LogregMulticlassModel(self.config, lr)
        )


class LogregMulticlassIpsAgent(ModelBasedAgent):
    """
    Logistic Regression Multiclass Agent (IPS version)
    """

    def __init__(self, config = Configuration(logreg_multiclass_ips_args)):
        super(LogregMulticlassIpsAgent, self).__init__(
            config,
            LogregMulticlassIpsModelBuilder(config)
        )
