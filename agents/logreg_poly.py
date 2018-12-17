from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

from agents import (
    ModelBasedAgent,
    AbstractFeatureProvider,
    ViewsFeaturesProvider,
    Model
)
from reco_gym import Configuration

logreg_poly_args = {
    'poly_degree': 2,

    'with_ips': False,
    'solver': 'lbfgs',
    'max_iter': 5000,
}


class LogregPolyModelBuilder(AbstractFeatureProvider):
    def __init__(self, config, weight_history_function = None):
        super(LogregPolyModelBuilder, self).__init__(config)
        self.weight_history_function = weight_history_function

    def build(self):
        class LogisticRegressionPolyFeaturesProvider(ViewsFeaturesProvider):
            """TBD"""

            def __init__(self, config, poly, poly_selection_flags):
                super(LogisticRegressionPolyFeaturesProvider, self).__init__(config)
                self.poly = poly
                self.poly_selection_flags = poly_selection_flags
                self.features_with_actions = np.zeros((self.config.num_products, 2 * self.config.num_products))
                ixs = np.array(range(self.config.num_products))
                self.features_with_actions[ixs, self.config.num_products + ixs] = 1

            def features(self):
                features_with_actions = self.features_with_actions.copy()
                features_with_actions[:, :self.config.num_products] = super().features()
                return self.poly.fit_transform(features_with_actions)[:, self.poly_selection_flags]

        class LogisticRegressionModel(Model):
            """TBD"""

            def __init__(self, config, logreg):
                super(LogisticRegressionModel, self).__init__(config)
                self.logreg = logreg

            def act(self, observation, features):
                action_proba = self.logreg.predict_proba(features)[:, 1]
                action = np.argmax(action_proba)
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': action_proba[action]
                    },
                }

        features, actions, deltas, pss = self.train_data()

        logreg = LogisticRegression(
            solver = self.config.solver,
            max_iter = self.config.max_iter,
            random_state = self.config.random_seed
        )

        action_vector = np.zeros((features.shape[0], self.config.num_products))
        action_vector[np.arange(features.shape[0]), actions] = 1

        features_with_actions = np.append(features, action_vector, axis = 1)

        poly = PolynomialFeatures(self.config.poly_degree)
        features_poly = poly.fit_transform(features_with_actions)

        only_first_degree = np.sum(poly.powers_, axis = 1) == 1
        only_with_actions = np.sum(poly.powers_[:, self.config.num_products:], axis = 1) == 1
        feature_selection_flags = only_first_degree | only_with_actions

        if self.config.with_ips:
            weights = np.minimum(deltas / pss, self.config.num_products)
            lr = logreg.fit(features_poly[:, feature_selection_flags], deltas, weights)
        else:
            lr = logreg.fit(features_poly[:, feature_selection_flags], deltas)

        return (
            LogisticRegressionPolyFeaturesProvider(self.config, poly, feature_selection_flags),
            LogisticRegressionModel(self.config, lr)
        )


class LogregPolyAgent(ModelBasedAgent):
    """
    TBD
    """

    def __init__(self, config = Configuration(logreg_poly_args)):
        super(LogregPolyAgent, self).__init__(
            config,
            LogregPolyModelBuilder(config)
        )
