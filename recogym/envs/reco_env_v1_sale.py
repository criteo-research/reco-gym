# Omega is the users latent representation of interests - vector of size K
#     omega is initialised when you have new user with reset
#     omega is updated at every timestep using timestep
#
# Gamma is the latent representation of organic products (matrix P by K)
# softmax(Gamma omega) is the next item probabilities (organic)

# Beta is the latent representation of response to actions (matrix P by K)
# sigmoid(beta omega) is the ctr for each action
import numpy as np
from numba import njit
from abc import ABC

import gym
import numpy as np
import pandas as pd
from gym.spaces import Discrete
from numpy.random.mtrand import RandomState
from scipy.special import expit as sigmoid
from tqdm import trange


from .abstract import AbstractEnv, env_args, f, organic, bandit, stop
# from .abstract_sale import AbstractEnvSale, env_args, organic
from .configuration import Configuration
from .context import DefaultContext
from .features.time import DefaultTimeGenerator
from .observation import Observation
from .session import OrganicSessions
from ..agents import Agent

# Default arguments for toy environment ------------------------------------

# inherit most arguments from abstract class
env_1_args = {
    **env_args,
    **{
        'K': 5,
        'sigma_omega_initial': 1,
        'sigma_omega': 0.1,
        'number_of_flips': 0,
        'sigma_mu_organic': 3,
        'change_omega_for_bandits': False,
        'normalize_beta': False
    }
}

##H
env_1_sale_args = {
    **env_1_args,
    **{
        'kappa': 0.5, # post-click scaling of theuser embedding update 
        'sigma_Lambda' : 1,
        # 'user_propensity' : {'a':2, 'b':6}, # propensity of buying of users, drawn from a beta distribution
        'psale_scale' : 0.25, # scaling coefficient for the probability of drawing a sale
        'delta_for_clicks' : 1
    }
}


@njit(nogil=True)
def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


# Maps behaviour into ctr - organic has real support ctr is on [0,1].
@njit(nogil=True)
def ff(xx, aa=5, bb=2, cc=0.3, dd=2, ee=6):
    # Magic numbers give a reasonable ctr of around 2%.
    return sig(aa * sig(bb * sig(cc * xx) - dd) - ee)


# Environment definition.
class RecoEnv1Sale(AbstractEnv): ##H

    def __init__(self):
        super(RecoEnv1Sale, self).__init__() ##H
        self.cached_state_seed = None
        self.proba_sales={} ##H
        self.proba_sales_after_scaling={} ##H
        self.user_ps_list=[] ##H


    def generate_organic_sessions(self):
        
        # Initialize session.
        session = OrganicSessions()
        sales = 0 ##H
        
        while self.state == organic:
            # Add next product view.
            self.update_product_view()
            session.next(
                DefaultContext(self.current_time, self.current_user_id),
                self.product_view
            )
            
            # Draw a sale and update total number of sales
            sale = self.draw_sale(self.product_view) ##H
            sales += sale ##H
            
            ##H
            if sale : 
                # self.state is not updated to stay in the organic state 
                # it is only updated in the sessions object
                self.current_time +=1 
                session.next(
                    DefaultContext(self.current_time, self.current_user_id),
                    self.product_view,
                    sale
                    )

            # Update markov state.
            self.update_state()

        return session, sales ##H

    def step(self, action_id):
        """

        Parameters
        ----------
        action_id : int between 1 and num_products indicating which
                 product recommended (aka which ad shown)

        Returns
        -------
        observation, reward, done, info : tuple
            observation (tuple) :
                a tuple of values (is_organic, product_view)
                is_organic - True  if Markov state is `organic`,
                             False if Markov state `bandit` or `stop`.
                product_view - if Markov state is `organic` then it is an int
                               between 1 and P where P is the number of
                               products otherwise it is None.
            reward (float) :
                if the previous state was
                    `bandit` - then reward is 1 if the user clicked on the ad
                               you recommended otherwise 0
                    `organic` - then reward is None
            done (bool) :
                whether it's time to reset the environment again.
                An episode is over at the end of a user's timeline (all of
                their organic and bandit sessions)
            info (dict) :
                 this is unused, it's always an empty dict
        """

        if self.first_step:
            assert (action_id is None)
            self.first_step = False
            sessions, sales = self.generate_organic_sessions() ##H
            return (
                Observation(
                    DefaultContext(
                        self.current_time,
                        self.current_user_id
                    ),
                    sessions
                ),
                0,  ##H or set back to none ?## or sales ? 
                self.state == stop,
                {}
            )

        assert (action_id is not None)
        # Calculate reward from action.
        click = self.draw_click(action_id)
        
        # Add whether the ad was clicked as info
        info = {'click' : click} ##H

        self.update_state()

        if click == 1:
            self.state = organic  # After a click, Organic Events always follow.
            self.update_user_feature(action_id) ##H # Update user feature based on reco

        # Markov state dependent logic.
        if self.state == organic:
            sessions, reward = self.generate_organic_sessions() ##H
        else:
            sessions = self.empty_sessions
            reward = 0


        return (
            Observation(
                DefaultContext(self.current_time, self.current_user_id),
                sessions
            ),
            reward, 
            self.state == stop,
            info
        )
    

    def step_offline(self, observation, reward, done):
        """Call step function wih the policy implemented by a particular Agent."""

        if self.first_step:
            action = None
        else:
            assert (hasattr(self, 'agent'))
            assert (observation is not None)
            if self.agent:
                action = self.agent.act(observation, reward, done)
            else:
                # Select a Product randomly.
                action = {
                    't': observation.context().time(),
                    'u': observation.context().user(),
                    'a': np.int16(self.rng.choice(self.config.num_products)),
                    'ps': 1.0 / self.config.num_products,
                    'ps-a': (
                        np.ones(self.config.num_products) / self.config.num_products
                        if self.config.with_ps_all else
                        ()
                    ),
                }

        if done:
            return (
                action,
                Observation(
                    DefaultContext(self.current_time, self.current_user_id),
                    self.empty_sessions
                ),
                0, 
                done, 
                {'click': 0} ##H
            )
        else:
            observation, reward, done, info = self.step(
                action['a'] if action is not None else None
            )

            return action, observation, reward, done, info

    def generate_logs(
            self,
            num_offline_users: int,
            agent: Agent = None,
            num_organic_offline_users: int = 0
    ):
        """
        Produce logs of applying an Agent in the Environment for the specified amount of Users.
        If the Agent is not provided, then the default Agent is used that randomly selects an Action.
        """

        if agent:
            old_agent = self.agent
            self.agent = agent

        data = {
            't': [],
            'u': [],
            'z': [],
            'v': [],
            'a': [],
            'c': [],
            'r': [],
            'ps': [],
            'ps-a': [],
        }

        def _store_organic(observation):
            assert (observation is not None)
            assert (observation.sessions() is not None)
            for session in observation.sessions():
                data['t'].append(session['t'])
                data['u'].append(session['u'])
                ##H before: just appending organic ##++ keep organic ?
                data['z'].append('organic' if session['z']=='pageview' else 'sale') 
                data['v'].append(session['v'])
                data['a'].append(None)
                data['c'].append(None)
                data['r'].append(None) ##H
                data['ps'].append(None)
                data['ps-a'].append(None)

        def _store_bandit(action, reward, info):
            if action:
                assert (reward is not None)
                data['t'].append(action['t'])
                data['u'].append(action['u'])
                data['z'].append('bandit')
                data['v'].append(None)
                data['a'].append(action['a'])
                data['c'].append(info['click']) ##H
                data['r'].append(reward)
                data['ps'].append(action['ps'])
                data['ps-a'].append(action['ps-a'] if 'ps-a' in action else ())

        unique_user_id = 0
        for _ in trange(num_organic_offline_users, desc='Organic Users'):
            self.reset(unique_user_id)
            unique_user_id += 1
            observation, _, _, _ = self.step(None)
            _store_organic(observation)

        for _ in trange(num_offline_users, desc='Users'):
            self.reset(unique_user_id)
            unique_user_id += 1
            observation, reward, done, _ = self.step(None)

            while not done:
                _store_organic(observation)
                action, observation, reward, done, info = self.step_offline(
                    observation, reward, done
                )
                _store_bandit(action, reward, info)

            _store_organic(observation)
            action, _, reward, done, info = self.step_offline(
                observation, reward, done
            )
            assert done, 'Done must not be changed!'
            _store_bandit(action, reward, info)

        data['t'] = np.array(data['t'], dtype=np.float32)
        data['u'] = pd.array(data['u'], dtype=pd.UInt16Dtype())
        data['v'] = pd.array(data['v'], dtype=pd.UInt16Dtype())
        data['a'] = pd.array(data['a'], dtype=pd.UInt16Dtype())
        data['c'] = np.array(data['c'], dtype=np.float32)

        if agent:
            self.agent = old_agent

        return pd.DataFrame().from_dict(data)
    



    def set_static_params(self):
        # Initialise the state transition matrix which is 3 by 3
        # high level transitions between organic, bandit and leave.
        self.state_transition = np.array([
            [0, self.config.prob_organic_to_bandit, self.config.prob_leave_organic],
            [self.config.prob_bandit_to_organic, 0, self.config.prob_leave_organic],
            [0.0, 0.0, 1.]
        ])

        self.state_transition[0, 0] = 1 - sum(self.state_transition[0, :])
        self.state_transition[1, 1] = 1 - sum(self.state_transition[1, :])

        # Initialise Gamma for all products (Organic).
        self.Gamma = self.rng.normal(
            size=(self.config.num_products, self.config.K)
        )
        # self.Gamma = np.abs(self.Gamma) ##H

        # Initialise mu_organic.
        self.mu_organic = self.rng.normal(
            0, self.config.sigma_mu_organic,
            size=(self.config.num_products, 1)
        )

        # Initialise beta, mu_bandit for all products (Bandit).
        self.generate_beta(self.config.number_of_flips)
        # self.beta = np.abs(self.beta) ##H

        ##H
        # Initialise sale embedding
        self.Lambda = self.rng.normal(
            scale = self.config.sigma_Lambda,
            size=(self.config.num_products, self.config.K)
        )
        # self.Lambda = np.abs(self.Lambda)
        

    # Create a new user.
    def reset(self, user_id=0):
        super().reset(user_id)
        ##++
        self.omega = self.rng.normal(
            0, self.config.sigma_omega_initial, size=(self.config.K, 1)
        )
        self.delta = self.omega #initialize additional term for user feature update
        self.proba_sales[self.current_user_id] = [] ##H
        self.proba_sales_after_scaling[self.current_user_id] = [] #HH
        # self.current_user_ps = self.rng.beta(
        #     a=self.config.user_propensity['a'], 
        #     b=self.config.user_propensity['b']
        # )
        self.current_user_ps = self.config.psale_scale
        self.user_ps_list.append(self.current_user_ps)

    # Update user state to one of (organic, bandit, leave) and their omega (latent factor).
    def update_state(self):
        old_state = self.state
        self.state = self.rng.choice(3, p=self.state_transition[self.state, :])
        assert (hasattr(self, 'time_generator'))
        old_time = self.current_time
        self.current_time = self.time_generator.new_time()
        time_delta = self.current_time - old_time
        omega_k = 1 if time_delta == 0 else time_delta

        # And update omega.
        ##++
        if self.config.change_omega_for_bandits or self.state == organic:
            self.omega = self.rng.normal(
                self.omega,
                self.config.sigma_omega * omega_k, size=(self.config.K, 1)
            )
        self.context_switch = old_state != self.state

    # Sample a click as response to recommendation when user in bandit state
    # click ~ Bernoulli().
    def draw_click(self, recommendation):
        # Personalised CTR for every recommended product.
        if self.config.change_omega_for_bandits or self.context_switch:
            if self.config.delta_for_clicks:
                user_feature = self.delta
            else :
                user_feature = self.omega
            self.cached_state_seed = (
                    self.beta @ user_feature + self.mu_bandit
            ).ravel()
        assert self.cached_state_seed is not None
        ctr = ff(self.cached_state_seed)
        click = self.rng.choice(
            [0, 1],
            p=[1 - ctr[recommendation], ctr[recommendation]]
        )
        return click

    # Sample the next organic product view.
    def update_product_view(self):
        log_uprob = (self.Gamma @ self.omega + self.mu_organic).ravel()
        log_uprob = log_uprob - max(log_uprob)
        uprob = np.exp(log_uprob)
        p=uprob / uprob.sum()
        self.product_view = np.int16(
            self.rng.choice(
                self.config.num_products,
                p=uprob / uprob.sum()
            )
        )

    def normalize_beta(self):
        self.beta = self.beta / np.sqrt((self.beta ** 2).sum(1)[:, np.newaxis])

    def generate_beta(self, number_of_flips):
        """Create Beta by flipping Gamma, but flips are between similar items only"""

        if number_of_flips == 0:
            self.beta = self.Gamma
            self.mu_bandit = self.mu_organic
            if self.config.normalize_beta:
                self.normalize_beta()

            return

        P, K = self.Gamma.shape
        index = np.arange(P)

        prod_cov = self.Gamma @ self.Gamma.T
        # We are always most correlated with ourselves so remove the diagonal.
        prod_cov = prod_cov - np.diag(np.diag(prod_cov))

        prod_cov_flat = prod_cov.flatten()

        already_used = set()
        flips = 0
        for p in prod_cov_flat.argsort()[::-1]:  # Find the most correlated entries
            # Convert flat indexes to 2d indexes
            ii, jj = int(p / P), np.mod(p, P)
            # Do flips between the most correlated entries
            # provided neither the row or col were used before.
            if not (ii in already_used or jj in already_used):
                index[ii] = jj  # Do a flip.
                index[jj] = ii
                already_used.add(ii)
                already_used.add(jj)
                flips += 1

                if flips == number_of_flips:
                    break

        self.beta = self.Gamma[index, :]
        self.mu_bandit = self.mu_organic[index, :]

        if self.config.normalize_beta:
            self.normalize_beta()
         
    ##H
    def draw_sale(self, a):
        ''' Draw sale following a Bernoulli'''
        # compute the sigmoid over the embeddings dot product
        p_sale = sig(self.Lambda[int(a),:] @ (self.omega+self.delta))[0]
        self.proba_sales[self.current_user_id].append(p_sale)
        
        # add the user propensity to buy (personnalized or generic)
        p_sale = self.current_user_ps * p_sale
        self.proba_sales_after_scaling[self.current_user_id].append(p_sale)
        
        sale = self.rng.choice(
            [0, 1],
            p=[1 - p_sale, p_sale]
        )
        return sale      
       
    ##H
    def update_user_feature(self,a):
        # #NB : scale new omega by the initial norm
        # omega_norm = np.linalg.norm(self.omega,ord=2)
        # add_term = np.expand_dims(self.config.kappa*self.Lambda[int(a),:], axis=1)
        # self.omega = (self.omega + add_term)/omega_norm
        self.delta = (1-self.config.kappa)*self.delta + self.config.kappa*np.expand_dims(self.Lambda[int(a),:], axis=1)

