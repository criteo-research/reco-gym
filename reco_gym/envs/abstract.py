
import gym
from gym.spaces import Discrete

from reco_gym import Organic_Session

from numpy import zeros
from numpy.random import RandomState, randint
from numpy.linalg import matrix_power
from scipy.special import expit as sigmoid
# change name of function since it's confusing

import pandas as pd

# Arguments shared between all environments ----------------------------------

env_args = {}

env_args['num_products'] = 10
env_args['num_users'] = 100

# Random Seed
env_args['random_seed'] = randint(2 ** 31 - 1)

# Markov State Transition Probabilities
env_args['prob_leave_bandit'] = 0.01
env_args['prob_leave_organic'] = 0.01
env_args['prob_bandit_to_organic'] = 0.05
env_args['prob_organic_to_bandit'] = 0.25


# Static function for squashing values between 0 and 1 ----------------------
def f(mat, offset=5):
    """monotonic increasing function as described in toy.pdf"""
    return sigmoid(mat - offset)


# Magic numbers for markov states -------------------------------------------
organic = 0
bandit = 1
stop = 2


# Environment definition ----------------------------------------------------
class AbstractEnv(gym.Env):

    def __init__(self):
        self.first_step = True

    def reset_random_seed(self):
        # Initialize Random State
        self.rng = RandomState(self.random_seed)

    def init_gym(self, args):

        # set all key word arguments as attributes
        for key in args:
            setattr(self, key, args[key])

        # Defining Action Space
        self.action_space = Discrete(self.num_products)

        # setting random seed for first time
        self.reset_random_seed()

        # setting any static parameters such as transition probabilities
        self.set_static_params()

        # set random seed for second time, ensures multiple epochs possible
        self.reset_random_seed()

    def reset(self):
        # Current state
        self.state = organic  # manually set first state as organic
        self.first_step = True

        # record number of times each product seen
        # for static policy calculation
        self.organic_views = zeros(self.num_products)

    def generate_organic_session(self):

        # Initialize session
        session = Organic_Session()

        while self.state == organic:
            # add next product view
            self.update_product_view()
            session.next(self.product_view)

            # update markov state
            self.update_state()

        return session

    def step(self, action):
        """

        Parameters
        ----------
        action : int between 1 and num_products indicating which
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
            assert(action is None)
            reward = None
        else:
            assert(action is not None)
            # Calculate reward from action
            reward = self.draw_click(action)

        # Markov state dependent logic
        if self.state == organic:
            observation = self.generate_organic_session()
        else:
            observation = None

        if reward != 1:
            # Update State
            self.update_state()
        else:
            self.state = organic  # clicks are followed by organic

        # Update done flag
        done = True if self.state == stop else False

        # No information to return
        info = {}

        self.first_step = False
        return observation, reward, done, info

    def step_offline(self):
        """calls step function but with fixed random policy"""

        if self.first_step:
            action = None
        else:
            action = self.rng.choice(self.num_products)

        observation, reward, done, info = self.step(action)

        return action, observation, reward, done, info

    # we need to think about if we need this
    def step_offline2(self):
        """calls step function but with fixed policy
           this policy randomly picks products in proportion to how much they
           have been viewed by all users"""

        if self.first_step:
            action = None
        else:
            # choosing action randomly in proportion with number of views
            prob = self.organic_views / sum(self.organic_views)
            action = self.rng.choice(self.num_products, p=prob)

        observation, reward, done, info = self.step(action)

        # adding organic session to organic view counts
        if observation is not None:
            for product in observation.get_views():
                self.organic_views[product] += 1

        return action, observation, reward, done, info

    def generate_data(self, num_offline_users):
        """Produce a DataFrame with the specified number of users"""

        user_id = 1
        data = list()
        for _ in range(num_offline_users):
            self.reset()
            observation, _, done, _ = self.step(None)

            d = {
                'v': [v[1] for v in observation],
                'u': user_id, 'r': -1, 'c': -1
            }
            data.append(pd.DataFrame().from_dict(d))

            while not done:
                action, observation, reward, done, info = self.step_offline()
                if done:
                    break
                if observation is not None:
                    d = {
                        'v': [v[1] for v in observation],
                        'u': user_id, 'r': -1, 'c': -1
                    }
                    data.append(pd.DataFrame().from_dict(d))

                data.append(
                    pd.DataFrame().from_dict(
                        {'v': -1, 'u': user_id, 'r': [action], 'c': reward}))

            user_id += 1
        return pd.concat(data)
