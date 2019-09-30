import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional as F

from recogym import Configuration, to_categorical
from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)

pytorch_blr_args = {
    'n_epochs': 50,
    'learning_rate': 0.005,
    'random_seed': np.random.randint(2 ** 31 - 1),
    'IPS': False
}


class PyTorchBLRModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(PyTorchBLRModelBuilder, self).__init__(config)

    def build(self):
        class PyTorchBLRFeaturesProvider(ViewsFeaturesProvider):
            """
            """

            def __init__(self, config):
                super(PyTorchBLRFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class PyTorchBLRModel(Model):
            """
            """

            def __init__(self, config, model):
                super(PyTorchBLRModel, self).__init__(config)
                self.model = model

            def act(self, observation, features):
                # OWN CODE
                X = features
                P = X.shape[1]
                A = np.eye(P)
                XA = np.kron(X, A)
                XA = Variable(torch.Tensor(XA))
                action_probs = self.model(XA).detach().numpy().ravel()

                action = np.argmax(action_probs)
                ps_all = np.zeros(P)
                ps_all[action] = 1.0
                # /OWN CODE

                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': 1.0,
                        'ps-a': ps_all,
                    },
                }

        class BinomialLogisticRegressionModel(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super(BinomialLogisticRegressionModel, self).__init__()
                # Generate weights - initialise randomly
                self.weight = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))
                torch.nn.init.kaiming_uniform_(self.weight, a = np.sqrt(5))
                # Calculcate bias - initialise randomly
                self.bias = torch.nn.Parameter(torch.Tensor(output_dim))
                fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / np.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

            def forward(self, x):
                # Compute linear transformation x.A.T
                pred = torch.sigmoid(F.linear(x, self.weight, self.bias))
                return pred

        # Get data
        features, actions, deltas, pss = self.train_data()

        # Extract data properly
        X = features
        N = X.shape[0]
        P = X.shape[1]
        A = to_categorical(actions, P)
        XA = np.array([np.kron(X[n, :], A[n, :]) for n in range(N)])
        y = deltas.reshape((deltas.shape[0], 1))

        # Generate model 
        model = BinomialLogisticRegressionModel(P * P, 1)

        # Convert data to torch objects
        X = Variable(torch.Tensor(XA))
        y = Variable(torch.Tensor(y))

        # Set sample weights
        w = torch.Tensor(pss.reshape((pss.shape[0], 1)) ** -1)

        # Binary cross-entropy for as loss criterion
        criterion = torch.nn.BCELoss(reduction = 'none')

        # LBFGS for optimisation
        optimiser = torch.optim.LBFGS(model.parameters(), lr = self.config.learning_rate)

        def closure():
            optimiser.zero_grad()
            p = model(X)
            loss = criterion(p, y)
            if self.config.IPS:
                loss *= w
            loss = loss.mean()
            loss.backward()
            return loss

        # Every epoch
        for epoch in range(self.config.n_epochs):
            # Reset gradients
            optimiser.zero_grad()
            # Forward pass
            p = model(X)
            # Compute loss
            loss = criterion(p, y)
            if self.config.IPS:
                loss *= w
            loss = loss.mean()
            # Backward pass
            loss.backward()
            optimiser.step(closure)

        return (
            PyTorchBLRFeaturesProvider(self.config),  # Poly is a bad name ..
            PyTorchBLRModel(self.config, model)
        )


class PyTorchBLRAgent(ModelBasedAgent):
    """
    PyTorch-based binomial logistic regression Agent.
    """

    def __init__(self, config = Configuration(pytorch_blr_args)):
        super(PyTorchBLRAgent, self).__init__(
            config,
            PyTorchBLRModelBuilder(config)
        )
