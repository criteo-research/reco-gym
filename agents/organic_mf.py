import torch
from torch import nn, optim, Tensor

# Default Arguments ----------------------------------------------------------

organic_mf_square_args = {}

organic_mf_square_args['num_products'] = 10
organic_mf_square_args['embed_dim'] = 5

organic_mf_square_args['mini_batch_size'] = 32

organic_mf_square_args['loss_function'] = nn.CrossEntropyLoss()
organic_mf_square_args['optim_function'] = optim.RMSprop
organic_mf_square_args['learning_rate'] = 0.01

# Model ----------------------------------------------------------------------
class OrganicMFSquare(nn.Module):
    def __init__(self, args):
        super().__init__()

        # set all key word arguments as attributes
        for key in args:
            setattr(self, key, args[key])

        self.product_embedding = nn.Embedding(
            self.num_products, self.embed_dim
        )

        self.output_layer = nn.Linear(
            self.embed_dim, self.num_products 
        )

        # initializing optimizer type
        self.optimizer = self.optim_function(
            self.parameters(), lr=self.learning_rate
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

            # no exploration strategy, choose maximum logit
            self.action = logits.argmax().item()

        return self.action

    def update_weights(self):
        """update weights of embedding matrices using mini batch of data"""
        # eliminate previous gradient
        self.optimizer.zero_grad()

        for prods in self.train_data:
            # calculating logit of action and last product viewed
            
            # Loop over the number of products
            for i in range(len(prods)-1):

                logit = self.forward(prods[i][-1])

                # converting label into Tensor
                label = Tensor([prods[i+1][-1]]).long()

                # calculating supervised loss
                loss = self.loss_function(logit, label)
                loss.backward()

        # update weight parameters
        self.optimizer.step()

    def train(self, observation, action, reward, done):
        """Method to deal with the """

        # increment step
        self.curr_step += 1

        # update weights of model once mini batch of data accumulated
        if self.curr_step % self.mini_batch_size == 0:
            self.update_weights()
            self.train_data = []
        else:
            if observation is not None:            
                data = (observation)
                self.train_data.append(data)
