import datetime

import gym
import pandas as pd
from recogym import (
    Configuration,
    env_1_args,
    gather_agent_stats,
    build_agent_init,
    AgentStats
)
from recogym.agents import OrganicUserEventCounterAgent, organic_user_count_args


def competition_score(
    num_products: int,
    num_users_to_train: int,
    num_users_to_score: int,
    random_seed: int,
    latent_factor: int,
    num_flips: int,
    log_epsilon: float,
    sigma_omega: float,
    agent_class,
    agent_configs
):
    TrainingDataSamples = tuple([num_users_to_train])
    TestingDataSamples = num_users_to_score
    StatEpochs = 1
    StatEpochsNewRandomSeed = True

    std_env_args = {
        **env_1_args,
        'random_seed': random_seed,
        'num_products': num_products,
        'K': latent_factor,
        'sigma_omega': sigma_omega,
        'number_of_flips': num_flips
    }

    env = gym.make('reco-gym-v1')

    time_start = datetime.datetime.now()
    agent_stats = gather_agent_stats(
        env,
        std_env_args,
        {
            'agent': OrganicUserEventCounterAgent(Configuration({
                **organic_user_count_args,
                **std_env_args,
                'select_randomly': True,
                'epsilon': log_epsilon,
                'num_products': num_products,
            })),
        },
        {
            **build_agent_init(
                'Test Agent',
                agent_class,
                {
                    **agent_configs,
                    'num_products': num_products,
                }
            ),
        },
        TrainingDataSamples,
        TestingDataSamples,
        StatEpochs,
        StatEpochsNewRandomSeed
    )

    q0_025 = []
    q0_500 = []
    q0_975 = []
    for agent_name in agent_stats[AgentStats.AGENTS]:
        agent_values = agent_stats[AgentStats.AGENTS][agent_name]
        q0_025.append(agent_values[AgentStats.Q0_025][0])
        q0_500.append(agent_values[AgentStats.Q0_500][0])
        q0_975.append(agent_values[AgentStats.Q0_975][0])

    time_end = datetime.datetime.now()
    seconds = (time_end - time_start).total_seconds()

    return pd.DataFrame(
        {
            'q0.025': q0_025,
            'q0.500': q0_500,
            'q0.975': q0_975,
            'time': [seconds],
        }
    )

