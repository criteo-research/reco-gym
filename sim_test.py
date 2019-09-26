import gym
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd
import argparse


from recogym.agents import OrganicUserEventCounterAgent, organic_user_count_args
from recogym import (
    Configuration,
    build_agent_init,
    env_1_args,
    gather_agent_stats,
    plot_agent_stats
)


parser = argparse.ArgumentParser()
parser.add_argument('--P', type=int, default=100, help='Number of products')
parser.add_argument('--U', type=int, default=100, help='Number of users to train on')
parser.add_argument('--Utest', type=int, default=1000, help='Number of users to test')
parser.add_argument('--seed', type=int, default=100, help='Seed')
parser.add_argument('--K', type=int, default=20, help='Number of latent factors')
parser.add_argument('--F', type=int, default=20, help='Number of flips, how different is bandit from organic')
parser.add_argument('--log_epsilon', type=float, default=0.05, help='Pop logging policy epsilon')
parser.add_argument('--sigma_omega', type=float, default=20, help='sigma_omega')
parser.add_argument('--entries_dir', type=str, default='entries', help='directory with agent .py files')



args = parser.parse_args()

P, U, Utest, seed, F, K, sigma_omega, log_epsilon, entries_dir = args.P, args.U, args.Utest, args.seed, args.F, args.K, args.sigma_omega, args.log_epsilon, args.entries_dir

print(args)

TrainingDataSamples = (U, )
TestingDataSamples = Utest
StatEpochs = 1
StatEpochsNewRandomSeed = True

std_env_args = {
    **env_1_args,
    'random_seed': seed,
    'num_products': P,
    'K': K,
    'sigma_omega': sigma_omega,
    'number_of_flips': F
}

env = gym.make('reco-gym-v1')

adf = []

for e in os.listdir(entries_dir):
    exec(open(entries_dir + '/' + e).read())
    print(open(entries_dir + '/' + e).read())

    start = datetime.datetime.now()

    # Gathering performance of agents for the logging policy: popularity based.
    agent_stats01 = gather_agent_stats(
        env,
        std_env_args,
        {
            'num_products': P,
            'number_of_flips': F,
            'agent': OrganicUserEventCounterAgent(Configuration({
                **organic_user_count_args,
                **std_env_args,
                'select_randomly': True,
                'epsilon': log_epsilon
            })),
        },
        {**agent,},
        TrainingDataSamples,
        TestingDataSamples,
        StatEpochs,
        StatEpochsNewRandomSeed
    )


    finish = datetime.datetime.now()

    mins = (finish - start).total_seconds() / 60.

    results = list(list(list(agent_stats01.values())[1].values())[0].values())

    df = pd.DataFrame({'q0.05':results[0], 'q0.5':results[1],'q0.95':results[2], 'time (mins)': [mins], 'entry': [e.replace('.py','')]})
    print(df)
    adf.append(df)

out_dir = entries_dir + '_' + str(P) + '_' + str(U) + '_' + str(Utest) + '_' + str(start)
os.mkdir(out_dir)
fp = open(out_dir + '/config.txt','w')
fp.write(str(args))
fp.close()

leaderboard = pd.concat(adf)
leaderboard = leaderboard.sort_values(by='q0.5',ascending=False)
leaderboard.to_csv(out_dir + '/leaderboard.csv')

