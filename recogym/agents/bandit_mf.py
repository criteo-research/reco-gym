import torch
import numpy as np
from torch import nn, optim, Tensor

from ..envs.configuration import Configuration

from .abstract import Agent

# Default Arguments.
bandit_mf_square_args = {
    'num_products': 10,
    'embed_dim': 5,
    'mini_batch_size': 32,
    'loss_function': nn.BCEWithLogitsLoss(),
    'optim_function': optim.RMSprop,
    'learning_rate': 0.01,
    'with_ps_all': False,
}


# Model.
class BanditMFSquare(nn.Module, Agent):
    def __init__(self, config = Configuration(bandit_mf_square_args)):
        nn.Module.__init__(self)
        Agent.__init__(self, config)

        self.product_embedding = nn.Embedding(
            self.config.num_products, self.config.embed_dim
        )
        self.user_embedding = nn.Embedding(
            self.config.num_products, self.config.embed_dim
        )

        # Initializing optimizer type.
        self.optimizer = self.config.optim_function(
            self.parameters(), lr = self.config.learning_rate
        )

        self.last_product_viewed = None
        self.curr_step = 0
        self.train_data = ([], [], [])
        self.all_products = np.arange(self.config.num_products)

    def forward(self, products, users = None):
        if users is None:
            users = np.full(products.shape[0], self.last_product_viewed)

        a = self.product_embedding(torch.LongTensor(products))
        b = self.user_embedding(torch.LongTensor(users))

        return torch.sum(a * b, dim = 1)

    def get_logits(self):
        """Returns vector of product recommendation logits"""
        return self.forward(self.all_products)

    def update_lpv(self, observation):
        """Updates the last product viewed based on the observation"""
        assert (observation is not None)
        assert (observation.sessions() is not None)
        if observation.sessions():
            self.last_product_viewed = observation.sessions()[-1]['v']

    def act(self, observation, reward, done):
        with torch.no_grad():
            # Update last product viewed.
            self.update_lpv(observation)

            # Get logits for all possible actions.
            logits = self.get_logits()

            # No exploration strategy, choose maximum logit.
            action = logits.argmax().item()
            if self.config.with_ps_all:
                all_ps = np.zeros(self.config.num_products)
                all_ps[action] = 1.0
            else:
                all_ps = ()

            return {
                **super().act(observation, reward, done),
                **{
                    'a': action,
                    'ps': logits[action],
                    'ps-a': all_ps,
                },
            }

    def update_weights(self):
        """Update weights of embedding matrices using mini batch of data"""
        if len(self.train_data[0]) != 0:
            # Eliminate previous gradient.
            self.optimizer.zero_grad()
            assert len(self.train_data[0]) == len(self.train_data[1])
            assert len(self.train_data[0]) == len(self.train_data[2])
            lpvs, actions, rewards = self.train_data

            # Calculating logit of action and last product viewed.
            logit = self.forward(np.array(actions), np.array(lpvs))

            # Converting reward into Tensor.
            reward = Tensor(np.array(rewards))

            # Calculating supervised loss.
            loss = self.config.loss_function(logit, reward)
            loss.backward()

            # Update weight parameters.
            self.optimizer.step()

    def train(self, observation, action, reward, done = False):
        # Update last product viewed.
        self.update_lpv(observation)

        # Increment step.
        self.curr_step += 1

        # Update weights of model once mini batch of data accumulated.
        if self.curr_step % self.config.mini_batch_size == 0:
            self.update_weights()
            self.train_data = ([], [], [])
        else:
            if action is not None and reward is not None:
                self.train_data[0].append(self.last_product_viewed)
                self.train_data[1].append(action['a'])
                self.train_data[2].append(reward)
