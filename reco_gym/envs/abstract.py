from abc import ABC

import gym
import pandas as pd
import numpy as np

from numpy.random.mtrand import RandomState
from scipy.special import expit as sigmoid
from gym.spaces import Discrete

from .session import OrganicSessions
from .context import DefaultContext
from .configuration import Configuration
from .observation import Observation

from .features.time import DefaultTimeGenerator

# Arguments shared between all environments.

env_args = {
    'num_products': 10,
    'num_users': 100,
    'random_seed': np.random.randint(2 ** 31 - 1),
    # Markov State Transition Probabilities.
    'prob_leave_bandit': 0.01,
    'prob_leave_organic': 0.01,
    'prob_bandit_to_organic': 0.05,
    'prob_organic_to_bandit': 0.25,
}


# Static function for squashing values between 0 and 1.
def f(mat, offset = 5):
    """Monotonic increasing function as described in toy.pdf."""
    return sigmoid(mat - offset)


# Magic numbers for Markov states.
organic = 0
bandit = 1
stop = 2


class AbstractEnv(gym.Env, ABC):

    def __init__(self):
        gym.Env.__init__(self)
        ABC.__init__(self)

        self.first_step = True
        self.config = None
        self.state = None
        self.current_user_id = None
        self.current_time = None
        self.empty_sessions = OrganicSessions()

    def reset_random_seed(self, epoch = 0):
        # Initialize Random State.
        assert (self.config.random_seed is not None)
        self.rng = RandomState(self.config.random_seed + epoch)

    def init_gym(self, args):

        self.config = Configuration(args)

        # Defining Action Space.
        self.action_space = Discrete(self.config.num_products)

        if 'time_generator' not in args:
            self.time_generator = DefaultTimeGenerator(self.config)
        else:
            self.time_generator = self.config.time_generator

        # Setting random seed for the first time.
        self.reset_random_seed()

        if 'agent' not in args:
            self.agent = None
        else:
            self.agent = self.config.agent

        # Setting any static parameters such as transition probabilities.
        self.set_static_params()

        # Set random seed for second time, ensures multiple epochs possible.
        self.reset_random_seed()

    def reset(self, user_id = 0):
        # Current state.
        self.first_step = True
        self.state = organic  # Manually set first state as Organic.

        self.time_generator.reset()
        if self.agent:
            self.agent.reset()

        self.current_time = self.time_generator.new_time()
        self.current_user_id = user_id

        # Record number of times each product seen for static policy calculation.
        self.organic_views = np.zeros(self.config.num_products)

    def generate_organic_sessions(self):

        # Initialize session.
        session = OrganicSessions()

        while self.state == organic:
            # Add next product view.
            self.update_product_view()
            session.next(
                DefaultContext(self.current_time, self.current_user_id),
                self.product_view
            )

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

        # No information to return.
        info = {}

        if self.first_step:
            assert (action_id is None)
            self.first_step = False
            sessions = self.generate_organic_sessions()
            return (
                Observation(
                    DefaultContext(
                        self.current_time,
                        self.current_user_id
                    ),
                    sessions
                ),
                None,
                None,
                info
            )

        assert (action_id is not None)
        # Calculate reward from action.
        reward = self.draw_click(action_id)

        self.update_state()

        if reward == 1:
            self.state = organic  # Clicks are followed by Organic.

        # Markov state dependent logic.
        if self.state == organic:
            sessions = self.generate_organic_sessions()
        else:
            sessions = self.empty_sessions

        # Update done flag.
        done = True if self.state == stop else False

        return (
            Observation(
                DefaultContext(self.current_time, self.current_user_id),
                sessions
            ),
            reward,
            done,
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
                    'a': int(self.rng.choice(self.config.num_products)),
                    'ps': 1.0 / self.config.num_products,
                    'ps-a': np.ones(self.config.num_products) / self.config.num_products,
                }

        observation, reward, done, info = self.step(action['a'] if action is not None else None)

        return action, observation, reward, done, info

    def generate_logs(self, num_offline_users, agent = None):
        """
        Produce logs of applying an Agent in the Environment for the specified amount of Users.
        If the Agent is not provided, then the default Agent is used that randomly selects an Action.
        """

        self.agent = agent

        data = {
            't': [],
            'u': [],
            'z': [],
            'v': [],
            'a': [],
            'c': [],
            'ps': [],
            'ps-a': [],
        }
        for user_id in range(num_offline_users):
            self.reset(user_id)
            observation, reward, done, _ = self.step(None)
            assert (observation is not None)
            assert (reward is None)

            while not done:
                assert (observation is not None)
                assert (observation.sessions() is not None)
                for session in observation.sessions():
                    data['t'].append(session['t'])
                    data['u'].append(session['u'])
                    data['z'].append('organic')
                    data['v'].append(session['v'])
                    data['a'].append(None)
                    data['c'].append(None)
                    data['ps'].append(None)
                    data['ps-a'].append(None)

                action, observation, reward, done, info = self.step_offline(observation, reward, done)

                data['t'].append(action['t'])
                data['u'].append(action['u'])
                data['z'].append('bandit')
                data['v'].append(None)
                data['a'].append(action['a'])
                data['c'].append(reward)
                data['ps'].append(action['ps'])
                data['ps-a'].append(action['ps-a'])

                if done:
                    break

        return pd.DataFrame().from_dict(data)
