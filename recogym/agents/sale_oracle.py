import numpy as np
from numba import njit

# from ..envs.configuration import Configuration
# from ..envs.reco_env_v1_sale import env_1_args
# from .abstract import Agent
from recogym.envs.reco_env_v1_sale import RecoEnv1Sale


@njit(nogil=True)
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

class SaleOracleAgent(Agent, RecoEnv1Sale):
    """
    Sale Oracle

    Has access to user and product features and popularity
    """

    def __init__(self, env):
        super(SaleOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done):
        """Make a recommendation"""
        self.delta = self.env.delta
        proba_difference = np.array([(self.delta[:,0]+self.Lambda[int(a),:]) @ self.Lambda[int(a),:] for a in range(self.env.config.num_products)])
#         print(np.max(proba_difference))
        action = np.argmax(proba_difference)
        self.list_actions.append(action)
        
        if self.env.config.with_ps_all:
            ps_all = np.zeros(self.config.num_products)
            ps_all[action] = 1.0
        else:
            ps_all = ()

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': self.env.config.psale_scale*sig(self.Lambda[int(action),:] @ (self.delta))[0],
                'ps-a': ps_all,
            },
        }

    def history(self):
        return self.list_actions

    def reset(self):
#         self.env.reset()
        self.list_actions = []
        self.delta = self.env.delta
        self.Lambda = self.env.Lambda
        self.list_actions = []


class ViewSaleOracleAgent(Agent, RecoEnv1Sale):
    """
    Sale Oracle

    Has access to user and product features and popularity
    """

    def __init__(self, env):
        super(ViewSaleOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done):
        """Make a recommendation"""
        self.delta = self.env.delta
        proba_view = np.array([self.user_feature_view @ self.Gamma[int(a),:] for a in range(self.env.config.num_products)])
        proba_difference = np.array([(self.delta[:,0]+self.Lambda[int(a),:]) @ self.Lambda[int(a),:] for a in range(self.env.config.num_products)])
        action = np.argmax(proba_view * proba_difference)
        self.list_actions.append(action)
        
        if self.env.config.with_ps_all:
            ps_all = np.zeros(self.config.num_products)
            ps_all[action] = 1.0
        else:
            ps_all = ()

        return {
            **super().act(observation, reward, done),
            **{
                'a': action,
                'ps': self.env.config.psale_scale*sig(self.Lambda[int(action),:] @ (self.delta))[0],
                'ps-a': ps_all,
            },
        }

    def history(self):
        return self.list_actions

    def reset(self):
        self.list_actions = []
        self.omega = self.env.omega
        self.delta = self.env.delta
        if "delta_for_views" in dir(self.env.config) is not None & self.env.config.delta_for_views == True :
            self.user_feature_view = self.delta
        else :
            self.user_feature_view = self.omega
        self.Lambda = self.env.Lambda
        self.Gamma = self.env.Gamma
        self.list_actions = []

