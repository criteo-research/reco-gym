import argparse

import pandas as pd
from challenge_entry import Entry
import gym, recogym
from recogym import env_1_args, Configuration
from copy import deepcopy

P = 100
U = 100
Utest = 100
seed = 42
F = 25
K = 20
sigma_omega = 0.05

def do_evaluation(P, U, Utest, seed, F, K, sigma_omega):
    env_1_args['random_seed'] = seed
    env_1_args['num_products'] = P
    env_1_args['number_of_flips'] = F
    env_1_args['sigma_omega'] = sigma_omega
    env_1_args['K'] = K


    env = gym.make('reco-gym-v1')
    env.init_gym(env_1_args)

    agent = Entry(Configuration({
        **{'num_products': P},
        **env_1_args,
    }))

    result = recogym.test_agent(deepcopy(env), deepcopy(agent), U, Utest)
    print(result)
    pd.DataFrame({'CTR': [result[0]], 'CTR_min':[result[1]], 'CTR_max':[result[2]]}).to_csv('result.csv')


if __name__ == "__main__":

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--P', type=int, default=100, help='Number of products')
    parser.add_argument('--U', type=int, default=100, help='Number of users to train on')
    parser.add_argument('--Utest', type=int, default=100, help='Number of users to test')
    parser.add_argument('--seed', type=int, default=100, help='Seed')
    parser.add_argument('--K', type=int, default=20, help='Number of latent factors')
    parser.add_argument('--sigma_omega', type=float, default=20, help='sigma_omega')

    args = parser.parse_args()
    do_evaluation(args.P, args.U, args.Utest, args.seed, args.F, args.K, args.sigma_omega)

