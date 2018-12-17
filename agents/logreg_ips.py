from sklearn.linear_model import LogisticRegression

import numpy as np
from agents import *
from reco_gym import Configuration

logreg_multiclass_ips_args = {
    'num_products': 10,
    'number_of_flips': 1,
    'random_seed': 42,

    'poly_degree': 2,
    'solver': 'lbfgs',
    'max_iter': 5000,
}


class LogregMulticlassIpsModelBuilder(AbstractFeatureProvider):
    """
    Logistic Regression Model Builder

    The class that provides both:
    * Logistic Regression Model
    * Feature Provider that builds a Feature Set suitable for the Logistic Regression Model
    """

    def __init__(self, config, weight_history_function = None):
        super(LogregMulticlassIpsModelBuilder, self).__init__(config)
        self.weight_history_function = weight_history_function

    def build(self):
        class LogregMulticlassViewsFeaturesProvider(ViewsFeaturesProvider):
            """TBD"""

            def __init__(self, config):
                super(LogregMulticlassViewsFeaturesProvider, self).__init__(config)

            def features(self):
                base_features = super().features()
                return base_features.reshape(1, self.config.num_products)

        class LogregMulticlassModel(Model):
            """TBD"""

            def __init__(self, config, logreg):
                super(LogregMulticlassModel, self).__init__(config)
                self.logreg = logreg

            def act(self, observation, features):
                action = self.logreg.predict(features).item()
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': self.logreg.predict_proba(features)[
                            0,
                            self.logreg.classes_ == action
                        ].item()
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
    TBD
    """

    def __init__(self, config = Configuration(logreg_multiclass_ips_args)):
        super(LogregMulticlassIpsAgent, self).__init__(
            config,
            LogregMulticlassIpsModelBuilder(config)
        )
