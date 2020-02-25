import math

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from tqdm import tqdm


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

    def train(self, observation, action, reward, done=False):
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

    def __init__(self, config, is_sparse=False):
        super(AbstractFeatureProvider, self).__init__(config)
        self.is_sparse = is_sparse

    def train_data(self):
        data = pd.DataFrame().from_dict(self.data)

        features = []
        actions = []
        pss = []
        deltas = []

        with_history = hasattr(self.config, 'weight_history_function')

        for user_id in tqdm(data['u'].unique(), desc='Train Data'):
            ix = 0
            ixs = []
            jxs = []
            if with_history:
                history = []

            assert data[data['u'] == user_id].shape[0] <= np.iinfo(np.int16).max

            for _, user_datum in data[data['u'] == user_id].iterrows():
                assert (not math.isnan(user_datum['t']))
                if user_datum['z'] == 'organic':
                    assert (math.isnan(user_datum['a']))
                    assert (math.isnan(user_datum['c']))
                    assert (not math.isnan(user_datum['v']))

                    view = np.int16(user_datum['v'])

                    if with_history:
                        ixs.append(np.int16(ix))
                        jxs.append(view)

                        history.append(np.int16(user_datum['t']))
                        ix += 1
                    else:
                        jxs.append(view)
                else:
                    assert (user_datum['z'] == 'bandit')
                    assert (not math.isnan(user_datum['a']))
                    assert (not math.isnan(user_datum['c']))
                    assert (math.isnan(user_datum['v']))

                    action = np.int16(user_datum['a'])
                    delta = np.int16(user_datum['c'])
                    ps = user_datum['ps']
                    time = np.int16(user_datum['t'])

                    if with_history:
                        assert len(ixs) == len(jxs)
                        views = sparse.coo_matrix(
                            (np.ones(len(ixs), dtype=np.int16), (ixs, jxs)),
                            shape=(len(ixs), self.config.num_products),
                            dtype=np.int16
                        )
                        weights = self.config.weight_history_function(
                            time - np.array(history)
                        )
                        weighted_views = views.multiply(weights[:, np.newaxis])
                        features.append(
                            sparse.coo_matrix(
                                weighted_views.sum(axis=0, dtype=np.float32),
                                copy=False
                            )
                        )
                    else:
                        views = sparse.coo_matrix(
                            (
                                np.ones(len(jxs), dtype=np.int16),
                                (np.zeros(len(jxs)), jxs)
                            ),
                            shape=(1, self.config.num_products),
                            dtype=np.int16
                        )
                        features.append(views)

                    actions.append(action)
                    deltas.append(delta)
                    pss.append(ps)

        out_features = sparse.vstack(features, format='csr')
        return (
            (
                out_features
                if self.is_sparse else
                np.array(out_features.todense(), dtype=np.float)
            ),
            np.array(actions, dtype=np.int16),
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

    def train(self, observation, action, reward, done=False):
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

    def __init__(self, config, is_sparse=False):
        super(ViewsFeaturesProvider, self).__init__(config)
        self.is_sparse = is_sparse
        self.with_history = (
                hasattr(self.config, 'weight_history_function')
                and
                self.config.weight_history_function is not None
        )
        self.reset()

    def observe(self, observation):
        assert (observation is not None)
        assert (observation.sessions() is not None)
        for session in observation.sessions():
            view = np.int16(session['v'])
            if self.with_history:
                self.ixs.append(np.int16(self.ix))
                self.jxs.append(view)
                self.history.append(np.int16(session['t']))
                self.ix += 1
            else:
                self.views[0, view] += 1

    def features(self, observation):
        if self.with_history:
            time = np.int16(observation.context().time())

            weights = self.config.weight_history_function(
                time - np.array(self.history)
            )
            weighted_views = self._views().multiply(weights[:, np.newaxis])
            views = sparse.coo_matrix(
                weighted_views.sum(axis=0, dtype=np.float32),
                copy=False
            )
            if self.is_sparse:
                return views
            else:
                return np.array(views.todense())
        else:
            return self._views()

    def reset(self):
        if self.with_history:
            self.ix = 0
            self.ixs = []
            self.jxs = []
            self.history = []
        else:
            if self.is_sparse:
                self.views = sparse.lil_matrix(
                    (1, self.config.num_products),
                    dtype=np.int16
                )
            else:
                self.views = np.zeros(
                    (1, self.config.num_products),
                    dtype=np.int16
                )

    def _views(self):
        if self.with_history:
            assert len(self.ixs) == len(self.jxs)
            return sparse.coo_matrix(
                (
                    np.ones(len(self.ixs), dtype=np.int16),
                    (self.ixs, self.jxs)
                ),
                shape=(len(self.ixs), self.config.num_products),
                dtype=np.int16
            )
        else:
            return self.views
