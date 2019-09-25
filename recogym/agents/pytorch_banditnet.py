import numpy as np
import tensorflow as tf
import torch
from collections import defaultdict
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.special import expit

from recogym.agents import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)
from recogym import Configuration

pytorch_banditnet_args = {
    'n_epochs': 50,
    'learning_rate': 0.005,
    'random_seed': np.random.randint(2 ** 31 - 1),
    'approximate_log_bound': True,
    'variance_penalisation_strength': .0,
    'lagrange_mult': 0.,
    'frac_neg': .0,
    'sample_smart': False     
}

class PyTorchBanditNetModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(PyTorchBanditNetModelBuilder, self).__init__(config)

    def build(self):
        class PyTorchBanditNetFeaturesProvider(ViewsFeaturesProvider):
            """
            """

            def __init__(self, config):
                super(PyTorchBanditNetFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class PyTorchBanditNetModel(Model):
            """
            """

            def __init__(self, config, model):
                super(PyTorchBanditNetModel, self).__init__(config)
                self.model = model

            def act(self, observation, features):
                # OWN CODE
                X = features
                P = X.shape[1]
                X = Variable(torch.Tensor(X))
                action_probs = self.model(X).detach().numpy().ravel()
                action = np.argmax(action_probs)
                ps_all = np.zeros(P)
                ps_all[action] = 1.0
                #/OWN CODE

                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': 1.0,
                        'ps-a': ps_all,
                    },
                }
        class MultinomialLogisticRegressionModel(torch.nn.Module):
            def __init__(self, input_dim, output_dim):
                super(MultinomialLogisticRegressionModel, self).__init__()
                # Generate weights - initialise randomly
                self.weight = torch.nn.Parameter(torch.Tensor(output_dim, input_dim))
                torch.nn.init.kaiming_uniform_(self.weight, a = np.sqrt(5))

            def forward(self, x):
                # Compute linear transformation x.A.T
                pred = F.linear(x, self.weight)
                return pred

        # Get data
        features, actions, deltas, pss = self.train_data()

        # Extract data properly
        X = features
        N = X.shape[0]
        P = X.shape[1]
        A = actions
        y = deltas

        # Generate model 
        model = MultinomialLogisticRegressionModel(P, P)

        #########  EXPERIMENTAL ##########
        pss = np.asarray(pss).reshape(-1,1)
        A = np.asarray(A).reshape(-1,1)
        # Get a mask for all clicked XA-pairs
        clicked = (y != 0)
        
        # Extract all clicks
        X1 = X[clicked]
        A1 = A[clicked]
        p1 = pss[clicked]
        print('Working with {0} clicks and {1} non-clicks...'.format(X1.shape[0], y.shape[0] - X1.shape[0]))

        if self.config.sample_smart:
            # Generate mapping from history-action pairs to the number of clicks received
            XA = np.hstack((X,A))
            hist2nclicks = defaultdict(int)
            for xa in XA[clicked]:
                # FIXME - hashing by making it a string is probably not that memory-efficient
                hist2nclicks[str(xa)] += 1

            # For every history-action-pair, add a column with the number of clicks they received
            C = np.fromiter((hist2nclicks[str(xa)] for xa in XA), np.int32).reshape(-1,1)
            XApC = np.hstack((XA,pss,C))

            # Now only keep XA-pairs that have never received clicks
            negXAp = XApC[XApC[:,-1] == 0][:,:-1]
            print('After filtering: {0} negative (X,A)-pairs left'.format(negXAp.shape[0]))

            # Extract non-clicks - sample from them
            negs = np.random.randint(0, negXAp.shape[0], int(self.config.frac_neg * X1.shape[0]))
            X0 = negXAp[negs,:-2]
            A0 = negXAp[negs, -2]
            p0 = negXAp[negs, -1]
            print('After sampling: {0} negative (X,A)-pairs left'.format(X0.shape[0]))
        else:
            # Sample from negatives uniformly at random
            n_not_clicked = X[~clicked].shape[0]
            negs = np.random.randint(0, n_not_clicked, int(self.config.frac_neg * X1.shape[0]))
            X0 = X[~clicked][negs]
            A0 = A[~clicked][negs]
            p0 = pss[~clicked][negs]
            print('After sampling: {0} negative (X,A)-pairs left'.format(X0.shape[0]))
        
        # Merge and shuffle
        order = np.random.permutation(X0.shape[0] + X1.shape[0])
        X = np.vstack((X0,X1))[order]
        A = np.concatenate((A0.ravel(),A1.ravel()))[order]
        p = np.concatenate((p0.ravel(),p1.ravel()))[order]
        deltas = np.concatenate((np.zeros(X0.shape[0]), np.ones(X1.shape[0])))[order]
        ######### /EXPERIMENTAL ##########
        
        # Convert data to torch objects - only clicks for learning
        X = Variable(torch.Tensor(X))
        A = Variable(torch.LongTensor(A))
        p0 = torch.Tensor(p)
        delta = Variable(torch.Tensor(deltas))
        delta_min_lambda = delta - self.config.lagrange_mult
        print("Min Delta-Lambda: {0}".format(np.min(delta_min_lambda.detach().numpy())))
        print("Mean Delta-Lambda: {0}".format(np.mean(delta_min_lambda.detach().numpy())))
        print("Max Delta-Lambda: {0}".format(np.max(delta_min_lambda.detach().numpy())))
        
        # LBFGS for optimisation
        optimiser = torch.optim.LBFGS(model.parameters(), lr = self.config.learning_rate)
        #optimiser = torch.optim.Adam(model.parameters())

        def closure():
            # Reset gradients
            optimiser.zero_grad()
            # Compute action predictions for clicks
            p = model(X)
            # Compute the loss
            if self.config.approximate_log_bound:
                # Optimise log of softmax
                loss = -torch.gather(F.log_softmax(p, dim = 1), 1, A.unsqueeze(1))
            else:
                # Directly optimise softmax
                loss = -torch.gather(F.softmax(p, dim = 1), 1, A.unsqueeze(1))
            # Multiply with (delta - lambda), divide with p0
            loss = delta_min_lambda * loss / p0
            # Compute the mean over all samples as the final loss
            loss = loss.mean()
        
            # If the variance regularisation strength is larger than zero
            if self.config.variance_penalisation_strength:
                # Get pi(a|x) for clicks
                prob_a = torch.gather(p, 1, A.unsqueeze(1))
                # Compute the expectation of the IPS estimate
                avg_weighted_reward = torch.mean(w * prob_a)
                # Compute the variance of the IPS estimate
                var = torch.sqrt(torch.sum(((w * prob_a) - avg_weighted_reward)**2) / (N - 1) / N)
                # Reweight with lambda and add to the loss
                loss = loss + self.config.variance_penalisation_strength * var

            # Backward pass
            loss.backward()
            return loss

        # Every epoch
        for epoch in range(self.config.n_epochs):
            # Optimisation step
            optimiser.step(closure)

        if torch.isnan(model.weight).any():
            print('------------------------------')
            print('PROBLEM: NaN weights detected!')
            print(model.weight)
            print('------------------------------')

        return (
            PyTorchBanditNetFeaturesProvider(self.config),
            PyTorchBanditNetModel(self.config, model)
        )

class PyTorchBanditNetAgent(ModelBasedAgent):
    """
    PyTorch-based multinomial logistic regression Agent.
    """

    def __init__(self, config = Configuration(pytorch_banditnet_args)):
        super(PyTorchBanditNetAgent, self).__init__(
            config,
            PyTorchBanditNetModelBuilder(config)
        )
