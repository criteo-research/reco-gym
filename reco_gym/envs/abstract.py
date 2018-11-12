import gym
import pandas as pd
import numpy as np

from gym.spaces import Discrete
from numpy.random.mtrand import RandomState

from reco_gym import Organic_Session
from scipy.special import expit as sigmoid

# Arguments shared between all environments.

env_args = {
    'num_products': 10,
    'num_users': 100,
    'random_seed': np.random.randint(2 ** 31 - 1),
    # Markov State Transition Probabilities.
    'prob_leave_bandit': 0.01,
    'prob_leave_organic': 0.01,
    'prob_bandit_to_organic': 0.05,
    'prob_organic_to_bandit': 0.25
}


# Static function for squashing values between 0 and 1.
def f(mat, offset = 5):
    """Monotonic increasing function as described in toy.pdf."""
    return sigmoid(mat - offset)


# Magic numbers for Markov states.
organic = 0
bandit = 1
stop = 2


# Define agent class.
class RandomAgent:
    def __init__(self, env):
        # Set Environment as an attribute of the Agent.
        self.env = env

    def act(self, observation, reward, done):
        """
        Act method returns an action based on a current observation and past history
        """
        prob = 1.0 / float(self.env['num_products'])
        action = np.random.choice(self.env['num_products'])

        return {
            'a': action,
            'ps': prob
        }

    def reset(self):
        pass


class AbstractEnv(gym.Env):

    def __init__(self):
        self.first_step = True

    def reset_random_seed(self):
        # Initialize Random State.
        self.rng = RandomState(self.random_seed)

    def init_gym(self, args):

        self.env = args

        # Set all key word arguments as attributes.
        for key in args:
            setattr(self, key, args[key])

        # Defining Action Space.
        self.action_space = Discrete(self.num_products)

        # Setting random seed for first time.
        self.reset_random_seed()

        # Setting any static parameters such as transition probabilities.
        self.set_static_params()

        # Set random seed for second time, ensures multiple epochs possible.
        self.reset_random_seed()

    def reset(self):
        # Current state.
        self.state = organic  # Manually set first state as Organic.
        self.first_step = True

        # Record number of times each product seen for static policy calculation.
        self.organic_views = np.zeros(self.num_products)

        if hasattr(self, 'agent'):
            self.agent.reset()

    def generate_organic_session(self):

        # Initialize session.
        session = Organic_Session()

        while self.state == organic:
            # Add next product view.
            self.update_product_view()
            session.next(self.product_view)

            # Update markov state.
            self.update_state()

        return session

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
            reward = None
        else:
            assert (action_id is not None)
            # Calculate reward from action.
            reward = self.draw_click(action_id)

        # Markov state dependent logic.
        if self.state == organic:
            observation = self.generate_organic_session()
        else:
            observation = None

        if reward != 1:
            # Update State.
            self.update_state()
        else:
            self.state = organic  # Clicks are followed by Organic.

        # Update done flag.
        done = True if self.state == stop else False

        # No information to return.
        info = {}

        self.first_step = False
        return observation, reward, done, info

    def step_offline(self, observation, reward, done):
        """Call step function wih the policy implemented by a particular Agent."""

        if self.first_step:
            action = None
        else:
            if hasattr(self, 'agent'):
                action = self.agent.act(observation, reward, done)
            else:
                action = {
                    'a': self.rng.choice(self.num_products),
                    'ps': 1.0 / self.num_products
                }

        observation, reward, done, info = self.step(action['a'] if action is not None else None)

        return action, observation, reward, done, info

    def generate_logs(self, num_offline_users, agent = None):
        """Produce a DataFrame with the specified number of users"""

        self.agent = RandomAgent(self.env) if agent is None else agent

        user_id = 1
        data = {
            'v': [],
            'u': [],
            'r': [],
            'c': [],
            'ps': [],
        }
        for _ in range(num_offline_users):
            self.reset()
            observation, reward, done, _ = self.step(None)

            for event in observation:
                data['v'].append(event[1])
                data['u'].append(user_id)
                data['r'].append(-1)
                data['c'].append(-1)
                data['ps'].append(None)

            while not done:
                action, observation, reward, done, info = self.step_offline(observation, reward, done)
                if done:
                    break

                if observation is not None:
                    for event in observation:
                        data['v'].append(event[1])
                        data['u'].append(user_id)
                        data['r'].append(-1)
                        data['c'].append(-1)
                        data['ps'].append(None)

                data['v'].append(-1)
                data['u'].append(user_id)
                data['r'].append(action['a'])
                data['c'].append(reward)
                data['ps'].append(action['ps'])

            user_id += 1
        return pd.DataFrame().from_dict(data)
