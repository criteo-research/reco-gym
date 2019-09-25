import numpy as np
import pandas as pd
import math


class Agent:
    """
    This is an abstract Agent class.
    The class defines an interface with methods those should be overwritten for a new Agent.
    """

    def __init__(self, config):
        self.config = config

    def act(self, observation, reward, done):
        """An act method takes in an observation, which could either be
           `None` or an Organic_Session (see recogym/session.py) and returns
           a integer between 0 and num_products indicating which product the
           agent recommends"""
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
        }

    def train(self, observation, action, reward, done = False):
        """Use this function to update your model based on observation, action,
            reward tuples"""
        pass

    def reset(self):
        pass


class ModelBuilder:
    """
    Model Builder

    The class that collects data obtained during a set of sessions
    (the data are collected via set of calls `train' method)
    and when it is decided that there is enough data to create a Model,
    the Model Builder generates BOTH the Feature Provider and the Model.
    Next time when the Model is to be used,
    the Feature Provider generates a Feature Set
    that is suitable for the Model.
    """

    def __init__(self, config):
        self.config = config
        self.data = None
        self.reset()

    def train(self, observation, action, reward, done):
        """
        Train a Model

        The method should be called every time when a new training data should be added.
        These data are used to train a Model.

        :param observation: Organic Sessions
        :param action: an Agent action for the Observation
        :param reward: reward (click/no click)
        :param done:

        :return: nothing
        """
        assert (observation is not None)
        assert (observation.sessions() is not None)
        for session in observation.sessions():
            self.data['t'].append(session['t'])
            self.data['u'].append(session['u'])
            self.data['z'].append('organic')
            self.data['v'].append(session['v'])
            self.data['a'].append(None)
            self.data['c'].append(None)
            self.data['ps'].append(None)

        if action:
            self.data['t'].append(action['t'])
            self.data['u'].append(action['u'])
            self.data['z'].append('bandit')
            self.data['v'].append(None)
            self.data['a'].append(action['a'])
            self.data['c'].append(reward)
            self.data['ps'].append(action['ps'])

    def build(self):
        """
        Build a Model

        The function generates a tuple: (FeatureProvider, Model)
        """
        raise NotImplemented

    def reset(self):
        """
        Reset Data

        The method clears all previously collected training data.
        """
        self.data = {
            't': [],
            'u': [],
            'z': [],
            'v': [],
            'a': [],
            'c': [],
            'ps': [],
        }


class Model:
    """
    Model

    """

    def __init__(self, config):
        self.config = config

    def act(self, observation, features):
        return {
            't': observation.context().time(),
            'u': observation.context().user(),
        }

    def reset(self):
        pass


class FeatureProvider:
    """
    Feature Provider

    The interface defines a set of methods used to:
    * collect data from which should be generated a Feature Set
    * generate a Feature Set suitable for a particular Model from previously collected data
    """

    def __init__(self, config):
        self.config = config

    def observe(self, observation):
        """
        Collect Observations

        The data are collected for a certain user

        :param observation:
        :return:
        """
        raise NotImplemented

    def features(self, observation):
        """
        Generate a Feature Set

        :return: a Feature Set suitable for a certain Model
        """
        raise NotImplemented

    def reset(self):
        """
        Reset

        Clear all previously collected data.
        :return: nothing
        """
        raise NotImplemented


class AbstractFeatureProvider(ModelBuilder):
    """
    Abstract Feature Provider

    The Feature Provider that contains the common logic in
    creation of a Feature Set that consists of:
    * Views (count of Organic Events: Products views)
    * Actions (Actions provided by an Agent)
    * Propensity Scores: probability of selecting an Action by an Agent
    * Delta (rewards: 1 -- there was a Click; 0 -- there was no)
    """

    def __init__(self, config):
        super(AbstractFeatureProvider, self).__init__(config)

    def train_data(self):
        data = pd.DataFrame().from_dict(self.data)

        features = []
        actions = []
        pss = []
        deltas = []

        for user_id in data['u'].unique():
            views = np.zeros((0, self.config.num_products))
            history = np.zeros((0, 1))
            for _, user_datum in data[data['u'] == user_id].iterrows():
                assert (not math.isnan(user_datum['t']))
                if user_datum['z'] == 'organic':
                    assert (math.isnan(user_datum['a']))
                    assert (math.isnan(user_datum['c']))
                    assert (not math.isnan(user_datum['v']))

                    view = int(user_datum['v'])

                    tmp_view = np.zeros(self.config.num_products)
                    tmp_view[view] = 1

                    # Append the latest view at the beginning of all views.
                    views = np.append(tmp_view[np.newaxis, :], views, axis = 0)
                    history = np.append(np.array([user_datum['t']])[np.newaxis, :], history, axis = 0)
                else:
                    assert (user_datum['z'] == 'bandit')
                    assert (not math.isnan(user_datum['a']))
                    assert (not math.isnan(user_datum['c']))
                    assert (math.isnan(user_datum['v']))

                    action = int(user_datum['a'])
                    delta = int(user_datum['c'])
                    ps = user_datum['ps']
                    time = user_datum['t']

                    if hasattr(self.config, 'weight_history_function'):
                        weights = self.config.weight_history_function(time - history)
                        train_views = views * weights
                    else:
                        train_views = views

                    feature = np.sum(train_views, axis = 0)

                    features.append(feature)
                    actions.append(action)
                    deltas.append(delta)
                    pss.append(ps)

        return (
            np.array(features),
            np.array(actions),
            np.array(deltas),
            np.array(pss)
        )


class ModelBasedAgent(Agent):
    """
    Model Based Agent

    This is a common implementation of the Agent that uses a certain Model when it acts.
    The Agent implements all routines needed to interact with the Model, namely:
    * training
    * acting
    """

    def __init__(self, config, model_builder):
        super(ModelBasedAgent, self).__init__(config)
        self.model_builder = model_builder
        self.feature_provider = None
        self.model = None

    def train(self, observation, action, reward, done = False):
        self.model_builder.train(observation, action, reward, done)

    def act(self, observation, reward, done):
        if self.model is None:
            assert (self.feature_provider is None)
            self.feature_provider, self.model = self.model_builder.build()
        self.feature_provider.observe(observation)
        return {
            **super().act(observation, reward, done),
            **self.model.act(observation, self.feature_provider.features(observation)),
        }

    def reset(self):
        if self.model is not None:
            assert (self.feature_provider is not None)
            self.feature_provider.reset()
            self.model.reset()


class ViewsFeaturesProvider(FeatureProvider):
    """
    Views Feature Provider

    This class provides Views of Products i.e. for all Products viewed by a users so far,
    the class returns a vector where at the index that corresponds to the Product you shall find
    amount of Views of that product.

    E.G.:
    Amount of Products is 5.
    Then, the class returns the vector [0, 3, 7, 0, 2].
    That means that Products with IDs were viewed the following amount of times:
        * 0 --> 0
        * 1 --> 3
        * 2 --> 7
        * 3 --> 0
        * 4 --> 2
    """

    def __init__(self, config):
        super(ViewsFeaturesProvider, self).__init__(config)
        self.history = None
        self.views = None
        self.reset()

    def observe(self, observation):
        assert (observation is not None)
        assert (observation.sessions() is not None)
        for session in observation.sessions():
            view = np.zeros((1, self.config.num_products))
            view[:, session['v']] = 1
            self.views = np.append(view, self.views, axis = 0)
            self.history = np.append(np.array([session['t']])[np.newaxis, :], self.history, axis = 0)

    def features(self, observation):
        if (
                hasattr(self.config, 'weight_history_function')
                and self.config.weight_history_function is not None
        ):
            time = observation.context().time()
            weights = self.config.weight_history_function(time - self.history)
            return np.sum(self.views * weights, axis = 0)
        else:
            return np.sum(self.views, axis = 0)

    def reset(self):
        self.views = np.zeros((0, self.config.num_products))
        self.history = np.zeros((0, 1))
