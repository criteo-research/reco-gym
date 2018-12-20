import gym

from reco_gym import env_0_args, env_1_args, test_agent
from copy import deepcopy

from reco_gym import Configuration

from agents import BanditMFSquare, bandit_mf_square_args
from agents import BanditCount, bandit_count_args
from agents import RandomAgent, random_args
from agents import LogregMulticlassIpsAgent, logreg_multiclass_ips_args
from agents import NnIpsAgent, nn_ips_args
from agents import OrganicCount, organic_count_args
from agents import OrganicUserEventCounterAgent, organic_user_count_args
from agents import LogregPolyAgent, logreg_poly_args

# Add a new environment here.
env_test = {
    "reco-gym-v1": env_1_args,
    "reco-gym-v0": env_0_args,
}

# Add a new agent here.
agent_test = {
    'prod2vec': BanditMFSquare(Configuration(bandit_mf_square_args)),
    'logistic': BanditCount(Configuration(bandit_count_args)),
    'randomagent': RandomAgent(Configuration(random_args)),
    'logreg_multiclass_ips': LogregMulticlassIpsAgent(Configuration(
        {
            **logreg_multiclass_ips_args,
            'select_randomly': False,
        })),
    'logreg_multiclass_ips R': LogregMulticlassIpsAgent(Configuration(
        {
            **logreg_multiclass_ips_args,
            'select_randomly': True,
        })),
    'nn_ips': NnIpsAgent(Configuration(nn_ips_args)),
    'organic_counter': OrganicCount(Configuration(organic_count_args)),
    'organic_user_counter': OrganicUserEventCounterAgent(Configuration(
        {
            **organic_user_count_args,
            'select_randomly': False,
        })),
    'organic_user_counter R': OrganicUserEventCounterAgent(Configuration(
        {
            **organic_user_count_args,
            'select_randomly': True,
        })),
    'logreg_poly': LogregPolyAgent(Configuration(logreg_poly_args))
}
eval_size = 5
organic_size = 5
samples = 200  # Set a big value to train model based on classifications.


def is_env_deterministic(env, users = 5):
    c_env = deepcopy(env)
    logs_a = c_env.generate_logs(users)
    c_env = deepcopy(env)
    logs_b = c_env.generate_logs(users)
    return logs_a.equals(logs_b)


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
