import multiprocessing
import time
from copy import deepcopy
import matplotlib.pyplot as plt
from multiprocessing import Pool

import reco_gym
from reco_gym import Configuration, TrainingApproach, EvolutionCase, AgentInit, AgentStats


def evolute_agent(
        env,
        agent,
        num_initial_train_users = 100,
        num_step_users = 1000,
        num_steps = 10,
        training_approach = TrainingApproach.ALL_DATA,
        sliding_window_samples = 10000):
    initial_agent = deepcopy(agent)

    unique_user_id = 0
    for u in range(num_initial_train_users):
        env.reset(unique_user_id + u)
        agent.reset()
        new_observation, reward, done, _ = env.step(None)
        while not done:
            old_observation = new_observation
            action, new_observation, reward, done, _ = env.step_offline(new_observation, reward, False)
            agent.train(old_observation, action, reward, done)
    unique_user_id += num_initial_train_users

    rewards = {
        EvolutionCase.SUCCESS: [],
        EvolutionCase.SUCCESS_GREEDY: [],
        EvolutionCase.FAILURE: [],
        EvolutionCase.FAILURE_GREEDY: [],
        EvolutionCase.ACTIONS: dict()
    }
    training_agent = deepcopy(agent)
    samples = 0

    for action_id in range(env.config.num_products):
        rewards[EvolutionCase.ACTIONS][action_id] = [0]

    for step in range(num_steps):
        successes = 0
        successes_greedy = 0
        failures = 0
        failures_greedy = 0

        for u in range(num_step_users):
            env.reset(unique_user_id + u)
            agent.reset()
            new_observation, reward, done, _ = env.step(None)
            while not done:
                old_observation = new_observation
                action = agent.act(old_observation, reward, done)
                new_observation, reward, done, info = env.step(action['a'])
                samples += 1

                should_update_training_data = False
                if training_approach == TrainingApproach.ALL_DATA or training_approach == TrainingApproach.LAST_STEP:
                    should_update_training_data = True
                elif training_approach == TrainingApproach.SLIDING_WINDOW_ALL_DATA:
                    should_update_training_data = samples % sliding_window_samples == 0
                elif training_approach == TrainingApproach.ALL_EXPLORATION_DATA:
                    should_update_training_data = not action['greedy']
                elif training_approach == TrainingApproach.SLIDING_WINDOW_EXPLORATION_DATA:
                    should_update_training_data = (not action['greedy']) and samples % sliding_window_samples == 0
                else:
                    assert False, f"Unknown Training Approach: {training_approach}"

                if should_update_training_data:
                    training_agent.train(old_observation, action, reward, done)

                if reward:
                    successes += 1
                    if 'greedy' in action and action['greedy']:
                        successes_greedy += 1
                    rewards[EvolutionCase.ACTIONS][action['a']][-1] += 1
                else:
                    if 'greedy' in action and action['greedy']:
                        failures_greedy += 1
                    failures += 1
        unique_user_id += num_step_users

        agent = training_agent
        for action_id in range(env.config.num_products):
            rewards[EvolutionCase.ACTIONS][action_id].append(0)

        if training_approach == TrainingApproach.LAST_STEP:
            training_agent = deepcopy(initial_agent)
        else:
            training_agent = deepcopy(agent)

        rewards[EvolutionCase.SUCCESS].append(successes)
        rewards[EvolutionCase.SUCCESS_GREEDY].append(successes_greedy)
        rewards[EvolutionCase.FAILURE].append(failures)
        rewards[EvolutionCase.FAILURE_GREEDY].append(failures_greedy)

    return rewards


def build_agent_init(agent_key, ctor, def_args):
    return {
        agent_key: {
            AgentInit.CTOR: ctor,
            AgentInit.DEF_ARGS: def_args,
        }
    }


def _collect_stats(args):
    """
    Function that is executed in a separate process.

    :param args: arguments of the process to be executed.

    :return: a vector of CTR for these confidence values:
        0th: Q0.500
        1st: Q0.025
        snd: Q0.975
    """
    start = time.time()
    print(f"Start: Num of Offline Users: {args['num_offline_users']}")
    stats = reco_gym.test_agent(
        deepcopy(args['env']),
        deepcopy(args['agent']),
        args['num_offline_users'],
        args['num_online_users'],
        args['num_organic_offline_users'],
        args['num_epochs'],
        args['epoch_with_random_reset']
    )
    print(f"End: Num of Offline Users: {args['num_offline_users']} ({time.time() - start}s)")
    return stats


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

        with Pool(processes = multiprocessing.cpu_count()) as pool:
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

            for result in pool.map(_collect_stats, argss) if num_epochs == 1 else [
                _collect_stats(args) for args in argss
            ]:
                stats[AgentStats.Q0_025].append(result[1])
                stats[AgentStats.Q0_500].append(result[0])
                stats[AgentStats.Q0_975].append(result[2])

        agent_stats[AgentStats.AGENTS][agent_key] = stats

    return agent_stats


def plot_agent_stats(agent_stats):
    _, ax = plt.subplots(
        1,
        1,
        figsize = (16, 8)
    )

    user_samples = agent_stats[AgentStats.SAMPLES]
    for agent_key in agent_stats[AgentStats.AGENTS]:
        stats = agent_stats[AgentStats.AGENTS][agent_key]

        ax.fill_between(
            user_samples,
            stats[AgentStats.Q0_975],
            stats[AgentStats.Q0_025],
            alpha = .05
        )

        ax.plot(user_samples, stats[AgentStats.Q0_500])

        ax.set_xlabel('Samples #')
        ax.set_ylabel('CTR')
        ax.legend([
            "$C^{CTR}_{0.5}$: " + f"{agent_key}" for agent_key in agent_stats[AgentStats.AGENTS]
        ])

    plt.show()
