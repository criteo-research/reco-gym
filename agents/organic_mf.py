from torch import nn, optim, Tensor

from .abstract import Agent
from reco_gym import Configuration

import numpy as np

# Default Arguments ----------------------------------------------------------
organic_mf_square_args = {
    'num_products': 10,
    'embed_dim': 5,
    'mini_batch_size': 32,
    'loss_function': nn.CrossEntropyLoss(),
    'optim_function': optim.RMSprop,
    'learning_rate': 0.01,
}


# Model ----------------------------------------------------------------------
class OrganicMFSquare(nn.Module, Agent):
    """
    Organic Matrix Factorisation (Square)

    The Agent that selects an Action from the model that performs
     Organic Events matrix factorisation.
    """

    def __init__(self, config = Configuration(organic_mf_square_args)):
        nn.Module.__init__(self)
        Agent.__init__(self, config)

        self.product_embedding = nn.Embedding(
            self.config.num_products, self.embed_dim
        )

        self.output_layer = nn.Linear(
            self.embed_dim, self.config.num_products
        )

        # Initializing optimizer type.
        self.optimizer = self.optim_function(
            self.parameters(), lr = self.learning_rate
        )

        self.last_product_viewed = None
        self.curr_step = 0
        self.train_data = []
        self.action = None

    def forward(self, product):

        product = Tensor([product]).long()

        a = self.product_embedding(product)
        b = self.output_layer(a)

        return b

    def act(self, observation, reward, done):

        if observation is not None:
            logits = self.forward(observation[-1][-1])

            # No exploration strategy, choose maximum logit.
            self.action = logits.argmax().item()

        all_ps = np.zeros(self.config.num_products)
        all_ps[self.action] = 1.0
        return {
            **super().act(observation, reward, done),
            **{
                'a': self.action,
                'ps': 1.0,
                'ps-a': all_ps,
            }
        }

    def update_weights(self):
        """Update weights of embedding matrices using mini batch of data"""
        # Eliminate previous gradient.
        self.optimizer.zero_grad()

        for prods in self.train_data:
            # Calculating logit of action and last product viewed.

            # Loop over the number of products.
            for i in range(len(prods) - 1):

                logit = self.forward(prods[i][-1])

                # Converting label into Tensor.
                label = Tensor([prods[i + 1][-1]]).long()

                # Calculating supervised loss.
                loss = self.config.loss_function(logit, label)
                loss.backward()

        # Update weight parameters.
        self.optimizer.step()

    def train(self, observation, action, reward, done = False):
        """Method to deal with the """

        # Increment step.
        self.curr_step += 1

        # Update weights of model once mini batch of data accumulated.
        if self.curr_step % self.config.mini_batch_size == 0:
            self.update_weights()
            self.train_data = []
        else:
            if observation is not None:
                data = observation
                self.train_data.append(data)
