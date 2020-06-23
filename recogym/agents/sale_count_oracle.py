import numpy as np
from numba import njit

# from ..envs.configuration import Configuration
from ..envs.reco_env_v1_sale import env_1_args, ff, sig
from .abstract import Agent
from recogym.envs.reco_env_v1_sale import RecoEnv1Sale


@njit(nogil=True)
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

class SaleCountOracleAgent(Agent, RecoEnv1Sale):
    """
    Sale Oracle

    Has access to user and product features and popularity
    """

    def __init__(self, env):
        super(SaleCountOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done):
        """Make a recommendation"""
        self.delta = self.env.delta
        embed = np.array([((1-self.kappa)*self.delta[:,0] + self.kappa*self.Lambda[int(a),:]) @ self.Lambda[int(a),:] for a in range(self.env.config.num_products)])
        action = np.argmax(embed)
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
        self.delta = self.env.delta
        self.Lambda = self.env.Lambda
        self.kappa = self.env.config.kappa
        self.list_actions = []


class ViewSaleCountOracleAgent(Agent, RecoEnv1Sale):
    """
    Sale Oracle

    Has access to user and product features and popularity
    """

    def __init__(self, env):
        super(ViewSaleCountOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done):
        """Make a recommendation"""
        self.omega = self.env.omega
        self.delta = self.env.delta
        if "delta_for_views" in dir(self.env.config) is not None & self.env.config.delta_for_views == True :
            self.user_feature_view = self.delta
        else :
            self.user_feature_view = self.omega
        
        # Proba of viewing each product
        log_proba_view = np.array([self.user_feature_view[:,0]@self.Gamma[int(a),:] + self.mu_organic[int(a)] for a in range(self.env.config.num_products)])
        proba_view = np.exp(log_proba_view - max(log_proba_view))
        proba_view = proba_view / proba_view.sum()
        proba_view = proba_view[:,0]
        
        # Difference in sale mean for each product if the product is recommended
        proba_with_click = np.array([sig(((1-self.kappa)*self.delta[:,0] + self.kappa*self.Lambda[int(a),:])@self.Lambda[int(a),:]) for a in range(self.env.config.num_products)])
        
        # Take argmax
        action = np.argmax(proba_view * proba_with_click)
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
        self.kappa = self.env.config.kappa
        self.omega = self.env.omega
        self.delta = self.env.delta
        if "delta_for_views" in dir(self.env.config) is not None & self.env.config.delta_for_views == True :
            self.user_feature_view = self.delta
        else :
            self.user_feature_view = self.omega
        self.Lambda = self.env.Lambda
        self.Gamma = self.env.Gamma
        self.mu_organic = self.env.mu_organic
        self.list_actions = []





class ClickViewSaleCountOracleAgent(Agent, RecoEnv1Sale):
    """
    Sale Oracle

    Has access to user and product features and popularity
    """

    def __init__(self, env):
        super(ClickViewSaleCountOracleAgent, self).__init__(env)
        self.env = env

    def act(self, observation, reward, done):
        """Make a recommendation"""
        self.omega = self.env.omega
        self.delta = self.env.delta
        
        if "delta_for_views" in dir(self.env.config) is not None & self.env.config.delta_for_views == True :
            self.user_feature_view = self.delta
        else :
            self.user_feature_view = self.omega
        
        if ("delta_for_clicks" in dir(self.env.config)) & (self.env.config.delta_for_clicks == 1):
            self.user_feature_click = self.delta
        else :
            self.user_feature_click = self.omega
        

        # Proba of clicking for each product
        proba_click = np.array([ff(self.user_feature_click[:,0]@self.beta[int(a),:] + self.mu_bandit[int(a)]) for a in range(self.env.config.num_products)])
        
        # Proba of viewing each product
        log_proba_view = np.array([self.user_feature_view[:,0]@self.Gamma[int(a),:] + self.mu_organic[int(a)] for a in range(self.env.config.num_products)])
        proba_view = np.exp(log_proba_view - max(log_proba_view))
        proba_view = proba_view / proba_view.sum()
        proba_view = proba_view[:,0]
        
        # Difference in sale mean for each product if the product is recommended
        proba_with_click = np.array([sig(((1-self.kappa)*self.delta[:,0] + self.kappa*self.Lambda[int(a),:])@self.Lambda[int(a),:]) for a in range(self.env.config.num_products)])
        
        # Take argmax
        action = np.argmax(proba_view * proba_with_click)
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
        self.kappa = self.env.config.kappa
        self.omega = self.env.omega
        self.delta = self.env.delta
        
        if "delta_for_views" in dir(self.env.config) is not None & self.env.config.delta_for_views == True :
            self.user_feature_view = self.delta
        else :
            self.user_feature_view = self.omega
        
        if ("delta_for_clicks" in dir(self.env.config)) & (self.env.config.delta_for_clicks ==1) :
            self.user_feature_click = self.delta
        else :
            self.user_feature_click = self.omega
        self.beta = self.env.beta
        self.Lambda = self.env.Lambda
        self.Gamma = self.env.Gamma
        self.mu_organic = self.env.mu_organic
        self.mu_bandit = self.env.mu_bandit
        self.list_actions = []

