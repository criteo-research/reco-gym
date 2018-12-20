import numpy as np
from numpy.random.mtrand import RandomState

from agents import AbstractFeatureProvider, ViewsFeaturesProvider, Model, ModelBasedAgent
from reco_gym import Configuration

organic_user_count_args = {
    'num_products': 10,

    # Select a Product randomly with the highest probability for the most frequently viewed product.
    'select_randomly': True,

    # Weight History Function: how treat each event back in time.
    'weight_history_function': None,

    'random_seed': 42,
}


class OrganicUserEventCounterModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(OrganicUserEventCounterModelBuilder, self).__init__(config)

    def build(self):

        class OrganicUserEventCounterModel(Model):
            """
            Organic Event Count Model (per a User).
            """

            def __init__(self, config):
                super(OrganicUserEventCounterModel, self).__init__(config)
                if config.select_randomly:
                    self.rng = RandomState(self.config.random_seed)

            def act(self, observation, features):
                action_proba = features / np.sum(features, axis = 0)
                if self.config.select_randomly:
                    action = self.rng.choice(self.config.num_products, p = action_proba)
                    ps = action_proba[action]
                else:
                    action = np.argmax(action_proba)
                    ps = 1.0
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': ps,
                    },
                }

        return (
            ViewsFeaturesProvider(self.config),
            OrganicUserEventCounterModel(self.config)
        )


class OrganicUserEventCounterAgent(ModelBasedAgent):
    """
    Organic Event Counter Agent

    The Agent that counts Organic views of Products (per a User)
    and selects an Action for the most frequently shown Product.
    """

    def __init__(self, config = Configuration(organic_user_count_args)):
        super(OrganicUserEventCounterAgent, self).__init__(
            config,
            OrganicUserEventCounterModelBuilder(config)
        )
