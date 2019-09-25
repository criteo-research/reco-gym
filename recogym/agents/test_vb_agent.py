import gym, recogym
import numpy as np
import pystan
import tensorflow as tf
from scipy.special import expit
from recogym import build_agent_init, plot_agent_stats
from recogym.agents import *
from copy import deepcopy
import time
from scipy.stats.distributions import beta

from abstract import (
    AbstractFeatureProvider,
    Model,
    ModelBasedAgent,
    ViewsFeaturesProvider
)
from recogym import Configuration

# env_0_args is a dictionary of default parameters (i.e. number of products)
from recogym import env_1_args, Configuration

# You can overwrite environment arguments here:
env_1_args['random_seed'] = 42

# Initialize the gym for the first time by calling .make() and .init_gym()
env = gym.make('reco-gym-v1')
env.init_gym(env_1_args)



def _collect_stats(args):
    env = args['env']
    agent = args['agent']
    num_offline_users = args['num_offline_users']
    num_online_users = args['num_online_users']
    num_organic_offline_users = args['num_organic_offline_users']
    epoch_with_random_reset = args['epoch_with_random_reset']
    epoch = args['epoch']

    start = time.time()
    print(f"Start: Agent Training #{epoch}")

    successes = 0
    failures = 0

    unique_user_id = 0
    new_agent = deepcopy(agent)

    if epoch_with_random_reset:
        env = deepcopy(env)
        env.reset_random_seed(epoch)

    # Offline organic Training.
    for u in range(num_organic_offline_users):
        env.reset(unique_user_id + u)
        unique_user_id += 1
        observation, _, _, _ = env.step(None)
        new_agent.train(observation, None, None, True)
    unique_user_id += num_organic_offline_users

    # Offline Training.
    for u in range(num_offline_users):
        env.reset(unique_user_id + u)
        new_observation, _, done, _ = env.step(None)
        while not done:
            old_observation = new_observation
            action, new_observation, reward, done, info = env.step_offline(old_observation, 0, False)
            new_agent.train(old_observation, action, reward, done)
    unique_user_id += num_offline_users

    # Online Testing.
    print(f"Start: Agent Testing #{epoch}")
    for u in range(num_online_users):
        env.reset(unique_user_id + u)
        new_agent.reset()
        new_observation, _, done, _ = env.step(None)
        reward = None
        done = None
        while not done:
            action = new_agent.act(new_observation, reward, done)
            new_observation, reward, done, info = env.step(action['a'])

            if reward:
                successes += 1
            else:
                failures += 1
    unique_user_id += num_online_users
    print(f"End: Agent Testing #{epoch} ({time.time() - start}s)")

    return {
        AgentStats.SUCCESSES: successes,
        AgentStats.FAILURES: failures,
    }

def test_agent(
        env,
        agent,
        num_offline_users = 1000,
        num_online_users = 100,
        num_organic_offline_users = 100,
        num_epochs = 1,
        epoch_with_random_reset = False
):
    successes = 0
    failures = 0

    #with Pool(processes = multiprocessing.cpu_count()) as pool:
    if True:
        argss = [
            {
                'env': env,
                'agent': agent,
                'num_offline_users': num_offline_users,
                'num_online_users': num_online_users,
                'num_organic_offline_users': num_organic_offline_users,
                'epoch_with_random_reset': epoch_with_random_reset,
                'epoch': epoch,
            }
            for epoch in range(num_epochs)
        ]

        for result in [_collect_stats(args) for args in argss] if num_epochs == 1 else map(_collect_stats, argss):
            successes += result[AgentStats.SUCCESSES]
            failures += result[AgentStats.FAILURES]

    return (
        beta.ppf(0.500, successes + 1, failures + 1),
        beta.ppf(0.025, successes + 1, failures + 1),
        beta.ppf(0.975, successes + 1, failures + 1)
    )


def build_agents(agents_init_data, new_env_args):
    agents = dict()
    for agent_key in agents_init_data:
        agent_init_data = agents_init_data[agent_key]
        ctor = agent_init_data[AgentInit.CTOR]
        def_args = agent_init_data[AgentInit.DEF_ARGS]
        agents[agent_key] = ctor(
            Configuration({
                **def_args,
                **new_env_args,
            })
        )
    return agents




def gather_agent_stats(
        env,
        env_args,
        extra_env_args,
        agents_init_data,
        user_samples = (100, 1000, 2000, 3000, 5000, 8000, 10000, 13000, 14000, 15000),
        num_online_users = 15000,
        num_epochs = 1,
        epoch_with_random_reset = False,
        num_organic_offline_users = 100
):
    """
    The function that gathers Agents statistics via evaluating Agent performance
     under different Environment conditions.

    :param env: the Environment where some changes should be introduced and where Agent stats should
        be gathered.
    :param env_args: Environment arguments (default ones).
    :param extra_env_args: extra Environment conditions those alter default values.
    :param agents_init_data: Agent initialisation data.
        This is a dictionary that has the following structure:
        {
            '<Agent Name>': {
                AgentInit.CTOR: <Constructor>,
                AgentInit.DEF_ARG: <Default Arguments>,
            }
        }
    :param user_samples: Number of Offline Users i.e. Users used to train a Model.
    :param num_online_users: Number of Online Users i.e. Users used to validate a Model.
    :param num_epochs: how many different epochs should be tried to gather stats?
    :param epoch_with_random_reset: should be a Random Seed reset at each new epoch?

    :return: a dictionary with stats
        {
            AgentStats.SAMPLES: [<vector of training offline users used to train a model>]
            AgentStats.AGENTS: {
                '<Agent Name>': {
                    AgentStats.Q0_025: [],
                    AgentStats.Q0_500: [],
                    AgentStats.Q0_975: [],
                }
            }
        }
    """
    new_env_args = {
        **env_args,
        **extra_env_args,
    }

    new_env = deepcopy(env)
    new_env.init_gym(new_env_args)

    agents = build_agents(agents_init_data, new_env_args)

    agent_stats = {
        AgentStats.SAMPLES: user_samples,
        AgentStats.AGENTS: dict(),
    }

    for agent_key in agents:
        print(f"Agent: {agent_key}")
        stats = {
            AgentStats.Q0_025: [],
            AgentStats.Q0_500: [],
            AgentStats.Q0_975: [],
        }

        #with Pool(processes = multiprocessing.cpu_count()) as pool:
        if True:
            argss = [
                {
                    'env': new_env,
                    'agent': agents[agent_key],
                    'num_offline_users': num_offline_users,
                    'num_online_users': num_online_users,
                    'num_organic_offline_users': num_organic_offline_users,
                    'num_epochs': num_epochs,
                    'epoch_with_random_reset': epoch_with_random_reset,
                }
                for num_offline_users in user_samples
            ]

            
            #for result in map(_collect_stats, argss) if num_epochs == 1 else [
            #    _collect_stats(args) for args in argss
            #]:


            for args in argss:            
                result = test_agent(deepcopy(args['env']), deepcopy(args['agent']), args['num_offline_users'], args['num_online_users'], args['num_organic_offline_users'], args['num_epochs'], args['epoch_with_random_reset'])

                stats[AgentStats.Q0_025].append(result[1])
                stats[AgentStats.Q0_500].append(result[0])
                stats[AgentStats.Q0_975].append(result[2])

        agent_stats[AgentStats.AGENTS][agent_key] = stats

    return agent_stats




bayesian_poly_args = {
    'num_products': 10,
    'random_seed': np.random.randint(2 ** 31 - 1),

    'poly_degree': 2,
    'max_iter': 5000,
    'aa': 1.,
    'bb': 1.
}

class BayesianModelBuilder(AbstractFeatureProvider):
    def __init__(self, config):
        super(BayesianModelBuilder, self).__init__(config)

    def build(self):
        class BayesianFeaturesProvider(ViewsFeaturesProvider):
            """
            """

            def __init__(self, config):
                super(BayesianFeaturesProvider, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class BayesianRegressionModel(Model):
            """
            """

            def __init__(self, config, Lambda):
                super(BayesianRegressionModel, self).__init__(config)
                self.Lambda = Lambda

            def act(self, observation, features):
                X = features
                P = X.shape[1]
                A = np.eye(P)
                XA = np.kron(X, A)

                action_proba = expit(np.matmul(XA, self.Lambda.T)).mean(1)
                action = np.argmax(action_proba)
                ps_all = np.zeros(self.config.num_products)
                ps_all[action] = 1.0
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': 1.0,
                        'ps-a': ps_all,
                    },
                }

        features, actions, deltas, pss = self.train_data()

        X = features
        N = X.shape[0]
        P = X.shape[1]
        A = tf.keras.utils.to_categorical(actions, P)
        XA = np.array([np.kron(X[n, :], A[n, :]) for n in range(N)])
        y = deltas  # clicks

        Sigma = np.kron(self.config.aa * np.eye(P) + self.config.bb,
                        self.config.aa * np.eye(P) + self.config.bb)
        fit = pystan.stan('model.stan', data = {
            'N': features.shape[0],
            'P': features.shape[1],
            'XA': XA,
            'y': deltas,
            'Sigma': Sigma
        }, chains = 1)
        s = fit.extract()
        Lambda = s['lambda']

        return (
            BayesianFeaturesProvider(self.config),  # Poly is a bad name ..
            BayesianRegressionModel(self.config, Lambda)
        )


class BayesianAgent(ModelBasedAgent):
    """
    Bayesian Agent.

    Note: the agent utilises Stan to train a model.
    """

    def __init__(self, config = Configuration(bayesian_poly_args)):
        super(BayesianAgent, self).__init__(
            config,
            BayesianModelBuilder(config)
        )




import pdb
from scipy import rand
from numpy.linalg import inv

def JJ(zeta):
    return 1./(2.*zeta)*(1./(1+np.exp(-zeta)) - 0.5)

def bayesian_logistic(Psi, y, mu_beta, Sigma_beta, iter=200):
    zeta = rand(Psi.shape[0])
    for _ in range(iter):
        q_Sigma = inv(inv(Sigma_beta) + 2 * np.matmul(np.matmul(Psi.T,np.diag(JJ(zeta))),Psi))
        q_mu = np.matmul(q_Sigma,(np.matmul(Psi.T, y-0.5) + np.matmul(inv(Sigma_beta), mu_beta)))
        zeta = np.sqrt(np.diag(np.matmul(np.matmul(Psi,q_Sigma + np.matmul(q_mu, q_mu.T)),Psi.T)))
    return q_mu, q_Sigma



from scipy.stats import multivariate_normal
class BayesianModelBuilderVB(AbstractFeatureProvider):
    def __init__(self, config):
        super(BayesianModelBuilderVB, self).__init__(config)

    def build(self):
        class BayesianFeaturesProviderVB(ViewsFeaturesProvider):
            """
            """

            def __init__(self, config):
                super(BayesianFeaturesProviderVB, self).__init__(config)

            def features(self, observation):
                base_features = super().features(observation)
                return base_features.reshape(1, self.config.num_products)

        class BayesianRegressionModelVB(Model):
            """
            """

            def __init__(self, config, Lambda):
                super(BayesianRegressionModelVB, self).__init__(config)
                self.Lambda = Lambda

            def act(self, observation, features):
                X = features
                P = X.shape[1]
                A = np.eye(P)
                XA = np.kron(X, A)

                action_proba = expit(np.matmul(XA, self.Lambda.T)).mean(1)
                action = np.argmax(action_proba)
                ps_all = np.zeros(self.config.num_products)
                ps_all[action] = 1.0
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': 1.0,
                        'ps-a': ps_all,
                    },
                }

        features, actions, deltas, pss = self.train_data()
        

        X = features
        N = X.shape[0]
        P = X.shape[1]
        A = tf.keras.utils.to_categorical(actions, P)
        XA = np.array([np.kron(X[n, :], A[n, :]) for n in range(N)])
        y = deltas  # clicks

        Sigma = np.kron(self.config.aa * np.eye(P) + self.config.bb,
                        self.config.aa * np.eye(P) + self.config.bb)



        q_mu, q_Sigma = bayesian_logistic(XA, y.reshape((N,1)), mu_beta = -6 * np.ones((P**2,1)) , Sigma_beta=Sigma)
        Lambda = multivariate_normal.rvs(q_mu.reshape(P**2),q_Sigma,1000)

        #stan version of the above (seems to agree well)
        #fit = pystan.stan('model.stan', data = {'N': features.shape[0], 'P': features.shape[1], 'XA': XA, 'y': y, 'Sigma': Sigma}, chains = 1)
        #s = fit.extract()
        #Lambda = s['lambda']

        ### 
        
        return (
            BayesianFeaturesProviderVB(self.config),  # Poly is a bad name ..
            BayesianRegressionModelVB(self.config, Lambda)
        )


class BayesianAgentVB(ModelBasedAgent):
    """
    Bayesian Agent.

    Note: the agent utilises VB to train a model.
    """

    def __init__(self, config = Configuration(bayesian_poly_args)):
        print('ffq')
        super(BayesianAgentVB, self).__init__(
            config,
            BayesianModelBuilderVB(config)
        )




from recogym import Configuration, TrainingApproach, EvolutionCase, AgentInit, AgentStats, RoiMetrics


if __name__ == "__main__":

    # Initialisation of different agents.
    agent_inits = {
    #    **build_agent_init(
    #        'Bayesian aa .01 bb 0.01',
    #        BayesianAgent,
    #        {
    #            **bayesian_poly_args,
    #            'aa': 0.01,
    #            'bb': 0.01
    #        }
    #    ),
        **build_agent_init(
            'BayesianVB aa .01 bb 0.01',
            BayesianAgentVB,
            {
                **bayesian_poly_args,
                'aa': 0.01,
                'bb': 0.01
            }
        )    
    }

    std_env_args = {
        **env_1_args,
        'random_seed': 44,
    }


    # Gathering performance of agents for the logging policy: popularity based.
    agent_stats01 = gather_agent_stats(
        env,
        std_env_args,
        {
            'num_products': 10,
            'number_of_flips': 5,
            'agent': OrganicUserEventCounterAgent(Configuration({
                **organic_user_count_args,
                **std_env_args,
                'select_randomly': True,
            })),
        },
        agent_inits,
        (50,),
        100,
        2,
        True
    )





