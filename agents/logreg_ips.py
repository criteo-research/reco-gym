from sklearn.linear_model import LogisticRegression

from agents import AbstractFeatureProvider, Model, ViewsFeaturesProvider, ModelBasedAgent
from reco_gym import Configuration


logreg_ips_args = {
    **{
        'num_products': 10,
        'sigma_omega': 1.0,
        'number_of_flips': 1,
        'random_seed': 42,

        'poly_degree': 2,
        'solver': 'lbfgs',
        'max_iter': 5000,
    }
}


class LogisticRegressionModelBuilder(AbstractFeatureProvider):
    """
    Logistic Regression Model Builder

    The class that provides both:
    * Logistic Regression Model
    * Feature Provider that builds a Feature Set suitable for the Logistic Regression Model
    """
    def __init__(self, config, weight_history_function = None):
        super(LogisticRegressionModelBuilder, self).__init__(config)
        self.weight_history_function = weight_history_function

    def build(self):
        class LogisticRegressionModel(Model):
            """TBD"""
            def __init__(self, config, logreg):
                super(LogisticRegressionModel, self).__init__(config)
                self.logreg = logreg

            def act(self, observation, features):
                action = self.logreg.predict(features)
                return {
                    **super().act(observation, features),
                    **{
                        'a': action[0],
                        'ps': self.logreg.predict_proba(features)[0, action]
                    },
                }

        class LogisticRegressionPolyFeaturesProvider(ViewsFeaturesProvider):
            """TBD"""
            def __init__(self, config, poly):
                super(LogisticRegressionPolyFeaturesProvider, self).__init__(config)
                self.poly = poly

            def features(self):
                base_features = super().features()
                return self.poly.fit_transform(base_features)

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
            ViewsFeaturesProvider(self.config),
            LogisticRegressionModel(self.config, lr)
        )


class LogregIpsAgent(ModelBasedAgent):
    """
    TBD
    """
    def __init__(self, config = Configuration(logreg_ips_args)):
        super(LogregIpsAgent, self).__init__(
            config,
            LogisticRegressionModelBuilder(config)
        )
