import torch
from torch import nn, optim, Tensor

# Default Arguments.

bandit_mf_square_args = {
    'num_products': 10,
    'embed_dim': 5,
    'mini_batch_size': 32,
    'loss_function': nn.BCEWithLogitsLoss(),
    'optim_function': optim.RMSprop,
    'learning_rate': 0.01
}


# Model.
class BanditMFSquare(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Set all key word arguments as attributes.
        for key in args:
            setattr(self, key, args[key])

        self.product_embedding = nn.Embedding(
            self.num_products, self.embed_dim
        )
        self.user_embedding = nn.Embedding(
            self.num_products, self.embed_dim
        )

        # Initializing optimizer type.
        self.optimizer = self.optim_function(
            self.parameters(), lr = self.learning_rate
        )

        self.last_product_viewed = None
        self.curr_step = 0
        self.train_data = []

    def forward(self, product, user = None):
        if user is None:
            user = self.last_product_viewed

        product = Tensor([product]).long()
        user = Tensor([user]).long()

        a = self.product_embedding(product).squeeze()
        b = self.user_embedding(user).squeeze()

        return torch.dot(a, b)

    def get_logits(self):
        """Returns vector of product recommendation logits"""
        logits = Tensor(self.num_products)

        for product in range(self.num_products):
            logits[product] = self.forward(product)

        return logits

    def update_lpv(self, observation):
        """Updates the last product viewed based on the observation"""
        if observation is not None:
            self.last_product_viewed = observation[-1][-1]

    def act(self, observation, reward, done):
        # Update last product viewed.
        self.update_lpv(observation)

        # Get logits for all possible actions.
        logits = self.get_logits()

        # No exploration strategy, choose maximum logit.
        action = logits.argmax().item()

        return {
            'a': action,
            'ps': logits[action],
        }

    def update_weights(self):
        """Update weights of embedding matrices using mini batch of data"""
        # Eliminate previous gradient.
        self.optimizer.zero_grad()

        for lpv, action, reward in self.train_data:
            # Calculating logit of action and last product viewed.
            logit = self.forward(action, lpv)

            # Converting reward into Tensor.
            reward = Tensor([reward]).squeeze()

            # Calculating supervised loss.
            loss = self.loss_function(logit, reward)
            loss.backward()

        # Update weight parameters.
        self.optimizer.step()

    def train(self, observation, action, reward, done):
        # Update last product viewed.
        self.update_lpv(observation)

        # Increment step.
        self.curr_step += 1

        # Update weights of model once mini batch of data accumulated.
        if self.curr_step % self.mini_batch_size == 0:
            self.update_weights()
            self.train_data = []
        else:
            if action is not None and reward is not None:
                data = (self.last_product_viewed, action['a'], reward)
                self.train_data.append(data)
