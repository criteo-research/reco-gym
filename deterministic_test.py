from copy import deepcopy

import gym
import numpy as np

from recogym import Configuration
from recogym import env_0_args, env_1_args, test_agent
from recogym.agents import BanditCount, bandit_count_args
from recogym.agents import BanditMFSquare, bandit_mf_square_args
from recogym.agents import LogregMulticlassIpsAgent, logreg_multiclass_ips_args
from recogym.agents import LogregPolyAgent, logreg_poly_args
from recogym.agents import NnIpsAgent, nn_ips_args
from recogym.agents import OrganicCount, organic_count_args
from recogym.agents import OrganicUserEventCounterAgent, organic_user_count_args
from recogym.agents import RandomAgent, random_args

# Add a new environment here.
env_test = {
    "reco-gym-v1": env_1_args,
    "reco-gym-v0": env_0_args,
}

RandomSeed = 42

# Add a new agent here.
agent_test = {
    'prod2vec': BanditMFSquare(Configuration(bandit_mf_square_args)),
    'logistic': BanditCount(Configuration(bandit_count_args)),
    'randomagent': RandomAgent(Configuration({
        **random_args,
        'random_seed': RandomSeed,
    })),
    'logreg_multiclass_ips': LogregMulticlassIpsAgent(Configuration({
        **logreg_multiclass_ips_args,
        'select_randomly': False,
    })),
    'logreg_multiclass_ips R': LogregMulticlassIpsAgent(Configuration({
        **logreg_multiclass_ips_args,
        'select_randomly': True,
        'random_seed': RandomSeed,
    })),
    'organic_counter': OrganicCount(Configuration(organic_count_args)),
    'organic_user_counter': OrganicUserEventCounterAgent(Configuration({
        **organic_user_count_args,
        'select_randomly': False,
    })),
    'organic_user_counter R': OrganicUserEventCounterAgent(Configuration({
        **organic_user_count_args,
        'select_randomly': True,
        'random_seed': RandomSeed,
    })),
    'logreg_poly': LogregPolyAgent(Configuration({
        **logreg_poly_args,
        'with_ips': False,
    })),
    'logreg_poly_ips': LogregPolyAgent(Configuration({
        **logreg_poly_args,
        'with_ips': True,
    })),
}
eval_size = 5
organic_size = 5
samples = 200  # Set a big value to train model based on classifications.


def is_env_deterministic(env, users=5):
    c_env = deepcopy(env)
    logs_a = c_env.generate_logs(num_offline_users=users, num_organic_offline_users=users)
    c_env = deepcopy(env)
    logs_b = c_env.generate_logs(num_offline_users=users, num_organic_offline_users=users)
    return np.mean(logs_a[~ np.isnan(logs_b.v)].v == logs_b[~ np.isnan(logs_b.v)].v) == 1.
    # return logs_a.equals(logs_b) # this can return false.. it isn't clear why atm ... most likely types differ..


if __name__ == "__main__":

    for env_name in env_test.keys():
        env = gym.make(env_name)
        env.init_gym(env_test[env_name])

        if is_env_deterministic(env):
            print(f"{env_name} appears deterministic")
        else:
            print(f"{env_name} is NOT deterministic")

        for agent_name in agent_test.keys():
            agent = agent_test[agent_name]
            a = test_agent(deepcopy(env), deepcopy(agent), samples, eval_size, organic_size)
            print(f"{agent_name} runs")
            b = test_agent(deepcopy(env), deepcopy(agent), samples, eval_size, organic_size)
            if a == b:
                print(f"{agent_name} appears deterministic")
            else:
                print(f"{agent_name} is NOT deterministic")
