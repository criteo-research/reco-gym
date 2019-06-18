import numpy as np
import torch
import torch.optim as optim

from numpy.random.mtrand import RandomState
from torch import nn

from recogym.agents import AbstractFeatureProvider, ViewsFeaturesProvider, Model, ModelBasedAgent
from recogym import Configuration

nn_ips_args = {
    'num_products': 10,
    'number_of_flips': 1,
    'random_seed': np.random.randint(2 ** 31 - 1),

    # Select a Product randomly with the the probability predicted by Neural Network with IPS.
    'select_randomly': False,

    'M': 111,
    'learning_rate': 0.01,
    'num_epochs': 100,
    'num_hidden': 20,
    'lambda_val': 0.01,
}


class IpsLoss(nn.Module):
    """
    IPS Loss Function
    """

    def __init__(self, config):
        super(IpsLoss, self).__init__()
        self.config = config
        self.clipping = nn.Threshold(self.config.M, self.config.M)

    def forward(self, hx, h0, deltas):
        u = self.clipping(hx / h0) * deltas
        return torch.mean(u) + self.config.lambda_val * torch.sqrt(torch.var(u) / deltas.shape[0])


class NeuralNet(nn.Module):
    """
    Neural Network Model

    This class implements a Neural Net model by using PyTorch.
    """

    def __init__(self, config):
        super(NeuralNet, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(self.config.num_products, self.config.num_hidden),
            nn.Sigmoid(),
            nn.Linear(self.config.num_hidden, self.config.num_hidden),
            nn.Sigmoid(),
            nn.Linear(self.config.num_hidden, self.config.num_products),
            nn.Softmax(dim = 1)
        )

    def forward(self, features):
        return self.model.forward(features)


class NnIpsModelBuilder(AbstractFeatureProvider):
    """
    Neural Net Inverse Propensity Score Model Builder
    """

    def __init__(self, config):
        super(NnIpsModelBuilder, self).__init__(config)

    def build(self):
        model = NeuralNet(self.config)
        criterion = IpsLoss(self.config)
        optimizer = optim.SGD(model.parameters(), lr = self.config.learning_rate)

        features, actions, deltas, pss = self.train_data()

        deltas = deltas[:, np.newaxis] * np.ones((1, self.config.num_products))
        pss = pss[:, np.newaxis] * np.ones((1, self.config.num_products))

        for epoch in range(self.config.num_epochs):
            optimizer.zero_grad()

            loss = criterion(
                model(torch.Tensor(features)),
                torch.Tensor(pss),
                torch.Tensor(-1.0 * deltas)
            )
            loss.backward()

            optimizer.step()

        class TorchFeatureProvider(ViewsFeaturesProvider):
            def __init__(self, config):
                super(TorchFeatureProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation).reshape(1, self.config.num_products)
                return torch.Tensor(base_features)

        class TorchModel(Model):
            def __init__(self, config, model):
                super(TorchModel, self).__init__(config)
                self.model = model
                if self.config.select_randomly:
                    self.rng = RandomState(self.config.random_seed)

            def act(self, observation, features):
                prob = self.model.forward(features)[0, :]
                if self.config.select_randomly:
                    prob = prob.detach().numpy()
                    action = self.rng.choice(
                        np.array(self.config.num_products),
                        p = prob
                    )
                    ps_all = prob
                else:
                    action = torch.argmax(prob).item()
                    ps_all = np.zeros(self.config.num_products)
                    ps_all[action] = 1.0
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': prob[action].item(),
                        'ps-a': ps_all,
                    },
                }

        return (
            TorchFeatureProvider(self.config),
            TorchModel(self.config, model)
        )


class NnIpsAgent(ModelBasedAgent):
    """
    Neural Network Agent with IPS
    """

    def __init__(self, config = Configuration(nn_ips_args)):
        super(NnIpsAgent, self).__init__(
            config,
            NnIpsModelBuilder(config)
        )
