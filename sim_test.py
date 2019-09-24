import gym
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd


from recogym.agents import OrganicUserEventCounterAgent, organic_user_count_args
from recogym import (
    Configuration,
    build_agent_init,
    env_1_args,
    gather_agent_stats,
    plot_agent_stats
)

RandomSeed = 42

TrainingDataSamples = (100, )
TestingDataSamples = 1000
StatEpochs = 1
StatEpochsNewRandomSeed = True

std_env_args = {
    **env_1_args,
    'random_seed': RandomSeed,
}

env = gym.make('reco-gym-v1')

adf = []

for e in os.listdir('entries'):
    exec(open('entries/' + e).read())

    # Initialisation of different agents.
    agent_inits = {
        **agent,
    }

    start = datetime.datetime.now()

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
                'epsilon': .5
            })),
        },
        agent_inits,
        TrainingDataSamples,
        TestingDataSamples,
        StatEpochs,
        StatEpochsNewRandomSeed
    )


    finish = datetime.datetime.now()

    mins = (finish - start).total_seconds() / 60.

    results = list(list(list(agent_stats01.values())[1].values())[0].values())

    df = pd.DataFrame({'q0.05':results[1], 'q0.5':results[0],'q0.95':results[2], 'time (mins)': [mins], 'entry': [e.replace('.py','')]})
    print(df)
    adf.append(df)

leaderboard = pd.concat(adf)
leaderboard = leaderboard.sort_values(by='q0.5',ascending=False)
leaderboard.to_csv('leaderboard.csv')

