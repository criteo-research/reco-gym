import multiprocessing
import time
from copy import deepcopy
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta

import recogym
from recogym import (
    AgentInit,
    AgentStats,
    Configuration,
    EvolutionCase,
    RoiMetrics,
    TrainingApproach
)
from recogym.agents import EpsilonGreedy, epsilon_greedy_args
from .envs.context import DefaultContext
from .envs.observation import Observation
from .envs.session import OrganicSessions

EpsilonDelta = .02
EpsilonSteps = 6  # Including epsilon = 0.0.
EpsilonPrecision = 2
EvolutionEpsilons = (0.00, 0.01, 0.02, 0.03, 0.05, 0.08)

GraphCTRMin = 0.009
GraphCTRMax = 0.021


# from Keras
def to_categorical(y, num_classes = None, dtype = 'float32'):
    y = np.array(y, dtype = 'int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype = dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def evaluate_agent(
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
            action, new_observation, reward, done, _ = env.step_offline(new_observation, reward,
                                                                        False)
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
                    should_update_training_data = (not action[
                        'greedy']) and samples % sliding_window_samples == 0
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
    stats = recogym.test_agent(
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

            for result in (
                    [_collect_stats(args) for args in argss]
                    if num_epochs == 1 else
                    pool.map(_collect_stats, argss)
            ):
                stats[AgentStats.Q0_025].append(result[1])
                stats[AgentStats.Q0_500].append(result[0])
                stats[AgentStats.Q0_975].append(result[2])

        agent_stats[AgentStats.AGENTS][agent_key] = stats

    return agent_stats


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


def generate_epsilons(epsilon_step = EpsilonDelta, iterations = EpsilonSteps):
    return [0.00, 0.01, 0.02, 0.03, 0.05, 0.08]


def format_epsilon(epsilon):
    return ("{0:." + f"{EpsilonPrecision}" + "f}").format(round(epsilon, EpsilonPrecision))


def _collect_evolution_stats(args):
    """
    Function that is executed in a separate process.

    :param args: arguments of the process to be executed.

    :return: a dictionary of Success/Failures of applying an Agent.
    """
    start = time.time()
    epsilon = args['epsilon']
    epsilon_key = format_epsilon(epsilon)
    print(f"Start: ε = {epsilon_key}")
    num_evolution_steps = args['num_evolution_steps']
    rewards = recogym.evaluate_agent(
        deepcopy(args['env']),
        args['agent'],
        args['num_initial_train_users'],
        args['num_step_users'],
        num_evolution_steps,
        args['training_approach']
    )

    assert (len(rewards[EvolutionCase.SUCCESS]) == len(rewards[EvolutionCase.FAILURE]))
    assert (len(rewards[EvolutionCase.SUCCESS]) == num_evolution_steps)
    print(f"End: ε = {epsilon_key} ({time.time() - start}s)")

    return {
        epsilon_key: {
            EvolutionCase.SUCCESS: rewards[EvolutionCase.SUCCESS],
            EvolutionCase.SUCCESS_GREEDY: rewards[EvolutionCase.SUCCESS_GREEDY],
            EvolutionCase.FAILURE: rewards[EvolutionCase.FAILURE],
            EvolutionCase.FAILURE_GREEDY: rewards[EvolutionCase.FAILURE_GREEDY],
            EvolutionCase.ACTIONS: rewards[EvolutionCase.ACTIONS]
        }
    }


def gather_exploration_stats(
        env,
        env_args,
        extra_env_args,
        agents_init_data,
        training_approach,
        num_initial_train_users = 1000,
        num_step_users = 1000,
        epsilons = EvolutionEpsilons,
        num_evolution_steps = 6
):
    """
    A helper function that collects data regarding Agents evolution
    under different values of epsilon for Epsilon-Greedy Selection Policy.

    :param env: The Environment where evolution should be applied;
         every time when a new step of the evolution is applied, the Environment is deeply copied
         thus the Environment does not interferes with evolution steps.

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


    :param training_approach:  A training approach applied in verification;
     for mode details look at `TrainingApproach' enum.

    :param num_initial_train_users: how many users' data should be used
     to train an initial model BEFORE evolution steps.

    :param num_step_users: how many users' data should be used
     at each evolution step.

     :param epsilons: a list of epsilon values.

    :param num_evolution_steps: how many evolution steps should be applied
     for an Agent with Epsilon-Greedy Selection Policy.

    :return a dictionary of Agent evolution statistics in the form:
        {
            'Agent Name': {
                'Epsilon Values': {
                    EvolutionCase.SUCCESS: [an array of clicks (for each ith step of evolution)]
                    EvolutionCase.FAILURE: [an array of failure to draw a click (for each ith step of evolution)]
                }
            }
        }
    """
    # A dictionary that stores all data of Agent evolution statistics.
    # Key is Agent Name, value is statistics.
    agent_evolution_stats = dict()

    new_env_args = {
        **env_args,
        **extra_env_args,
    }

    new_env = deepcopy(env)
    new_env.init_gym(new_env_args)

    agents = build_agents(agents_init_data, new_env_args)

    for agent_key in agents:
        print(f"Agent: {agent_key}")
        agent_stats = dict()

        with Pool(processes = multiprocessing.cpu_count()) as pool:
            for result in pool.map(
                    _collect_evolution_stats,
                    [
                        {
                            'epsilon': epsilon,
                            'env': new_env,
                            'agent': EpsilonGreedy(
                                Configuration({
                                    **epsilon_greedy_args,
                                    **new_env_args,
                                    'epsilon': epsilon,
                                }),
                                deepcopy(agents[agent_key])
                            ),
                            'num_initial_train_users': num_initial_train_users,
                            'num_step_users': num_step_users,
                            'num_evolution_steps': num_evolution_steps,
                            'training_approach': training_approach,
                        }
                        for epsilon in epsilons
                    ]
            ):
                agent_stats = {
                    **agent_stats,
                    **result,
                }

        agent_evolution_stats[agent_key] = agent_stats

    return agent_evolution_stats


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


def plot_evolution_stats(
        agent_evolution_stats,
        max_agents_per_row = 2,
        epsilons = EvolutionEpsilons,
        plot_min = GraphCTRMin,
        plot_max = GraphCTRMax
):
    figs, axs = plt.subplots(
        int(len(agent_evolution_stats) / max_agents_per_row),
        max_agents_per_row,
        figsize = (16, 10),
        squeeze = False
    )
    labels = [("$\epsilon=$" + format_epsilon(epsilon)) for epsilon in epsilons]

    for (ix, agent_key) in enumerate(agent_evolution_stats):
        ax = axs[int(ix / max_agents_per_row), int(ix % max_agents_per_row)]
        agent_evolution_stat = agent_evolution_stats[agent_key]

        ctr_means = []
        for epsilon in epsilons:
            epsilon_key = format_epsilon(epsilon)
            evolution_stat = agent_evolution_stat[epsilon_key]

            steps = []
            ms = []
            q0_025 = []
            q0_975 = []

            assert (len(evolution_stat[EvolutionCase.SUCCESS]) == len(
                evolution_stat[EvolutionCase.FAILURE]))
            for step in range(len(evolution_stat[EvolutionCase.SUCCESS])):
                steps.append(step)
                successes = evolution_stat[EvolutionCase.SUCCESS][step]
                failures = evolution_stat[EvolutionCase.FAILURE][step]

                ms.append(beta.ppf(0.5, successes + 1, failures + 1))
                q0_025.append(beta.ppf(0.025, successes + 1, failures + 1))
                q0_975.append(beta.ppf(0.975, successes + 1, failures + 1))

            ctr_means.append(np.mean(ms))

            ax.fill_between(
                range(len(steps)),
                q0_975,
                q0_025,
                alpha = .05
            )
            ax.plot(steps, ms)

        ctr_means_mean = np.mean(ctr_means)
        ctr_means_div = np.sqrt(np.var(ctr_means))
        ax.set_title(
            f"Agent: {agent_key}\n"
            + "$\hat{Q}^{CTR}_{0.5}="
            + "{0:.5f}".format(round(ctr_means_mean, 5))
            + "$, "
            + "$\hat{\sigma}^{CTR}_{0.5}="
            + "{0:.5f}".format(round(ctr_means_div, 5))
            + "$"
        )
        ax.legend(labels)
        ax.set_ylabel('CTR')
        ax.set_ylim([plot_min, plot_max])

    plt.subplots_adjust(hspace = .5)
    plt.show()


def plot_heat_actions(
        agent_evolution_stats,
        epsilons = EvolutionEpsilons
):
    max_epsilons_per_row = len(epsilons)
    the_first_agent = next(iter(agent_evolution_stats.values()))
    epsilon_steps = len(the_first_agent)
    rows = int(len(agent_evolution_stats) * epsilon_steps / max_epsilons_per_row)
    figs, axs = plt.subplots(
        int(len(agent_evolution_stats) * epsilon_steps / max_epsilons_per_row),
        max_epsilons_per_row,
        figsize = (16, 4 * rows),
        squeeze = False
    )

    for (ix, agent_key) in enumerate(agent_evolution_stats):
        agent_evolution_stat = agent_evolution_stats[agent_key]
        for (jx, epsilon_key) in enumerate(agent_evolution_stat):
            flat_index = ix * epsilon_steps + jx
            ax = axs[int(flat_index / max_epsilons_per_row), int(flat_index % max_epsilons_per_row)]

            evolution_stat = agent_evolution_stat[epsilon_key]

            action_stats = evolution_stat[EvolutionCase.ACTIONS]
            total_actions = len(action_stats)
            heat_data = []
            for kx in range(total_actions):
                heat_data.append(action_stats[kx])

            heat_data = np.array(heat_data)
            im = ax.imshow(heat_data)

            ax.set_yticks(np.arange(total_actions))
            ax.set_yticklabels([f"{action_id}" for action_id in range(total_actions)])

            ax.set_title(f"Agent: {agent_key}\n$\epsilon=${epsilon_key}")

            _ = ax.figure.colorbar(im, ax = ax)

    plt.show()


def plot_roi(
        agent_evolution_stats,
        epsilons = EvolutionEpsilons,
        max_agents_per_row = 2
):
    """
    A helper function that calculates Return of Investment (ROI) for applying Epsilon-Greedy Selection Policy.

    :param agent_evolution_stats: statistic about Agent evolution collected in `build_exploration_data'.

    :param epsilons: a list of epsilon values.

    :param max_agents_per_row: how many graphs should be drawn per a row

    :return: a dictionary of Agent ROI after applying Epsilon-Greedy Selection Strategy in the following form:
        {
            'Agent Name': {
                'Epsilon Value': {
                    Metrics.ROI: [an array of ROIs for each ith step (starting from 1st step)]
                }
            }
        }
    """
    figs, axs = plt.subplots(
        int(len(agent_evolution_stats) / max_agents_per_row),
        max_agents_per_row,
        figsize = (16, 8),
        squeeze = False
    )
    labels = [("$\epsilon=$" + format_epsilon(epsilon)) for epsilon in epsilons if epsilon != 0.0]

    agent_roi_stats = dict()

    for (ix, agent_key) in enumerate(agent_evolution_stats):
        ax = axs[int(ix / max_agents_per_row), int(ix % max_agents_per_row)]
        agent_stat = agent_evolution_stats[agent_key]
        zero_epsilon_key = format_epsilon(0)
        zero_epsilon = agent_stat[zero_epsilon_key]
        zero_success_evolutions = zero_epsilon[EvolutionCase.SUCCESS]
        zero_failure_evolutions = zero_epsilon[EvolutionCase.FAILURE]
        assert (len(zero_success_evolutions))

        agent_stats = dict()
        roi_mean_means = []
        for epsilon in generate_epsilons():
            if zero_epsilon_key == format_epsilon(epsilon):
                continue

            epsilon_key = format_epsilon(epsilon)
            agent_stats[epsilon_key] = {
                RoiMetrics.ROI_0_025: [],
                RoiMetrics.ROI_MEAN: [],
                RoiMetrics.ROI_0_975: [],
            }
            epsilon_evolutions = agent_stat[epsilon_key]
            success_greedy_evolutions = epsilon_evolutions[EvolutionCase.SUCCESS_GREEDY]
            failure_greedy_evolutions = epsilon_evolutions[EvolutionCase.FAILURE_GREEDY]
            assert (len(success_greedy_evolutions) == len(failure_greedy_evolutions))
            assert (len(zero_success_evolutions) == len(success_greedy_evolutions))
            steps = []
            roi_means = []
            for step in range(1, len(epsilon_evolutions[EvolutionCase.SUCCESS])):
                previous_zero_successes = zero_success_evolutions[step - 1]
                previous_zero_failures = zero_failure_evolutions[step - 1]
                current_zero_successes = zero_success_evolutions[step]
                current_zero_failures = zero_failure_evolutions[step]
                current_epsilon_greedy_successes = success_greedy_evolutions[step]
                current_epsilon_greedy_failures = failure_greedy_evolutions[step]

                def roi_with_confidence_interval(
                        epsilon,
                        previous_zero_successes,
                        previous_zero_failures,
                        current_zero_successes,
                        current_zero_failures,
                        current_epsilon_greedy_successes,
                        current_epsilon_greedy_failures
                ):
                    def roi_formulae(
                            epsilon,
                            previous_zero,
                            current_zero,
                            current_epsilon_greedy
                    ):
                        current_gain = current_epsilon_greedy / (1 - epsilon) - current_zero
                        roi = current_gain / (epsilon * previous_zero)
                        return roi

                    return {
                        RoiMetrics.ROI_SUCCESS: roi_formulae(
                            epsilon,
                            previous_zero_successes,
                            current_zero_successes,
                            current_epsilon_greedy_successes
                        ),
                        RoiMetrics.ROI_FAILURE: roi_formulae(
                            epsilon,
                            previous_zero_failures,
                            current_zero_failures,
                            current_epsilon_greedy_failures
                        )
                    }

                roi_mean = roi_with_confidence_interval(
                    epsilon,
                    previous_zero_successes,
                    previous_zero_failures,
                    current_zero_successes,
                    current_zero_failures,
                    current_epsilon_greedy_successes,
                    current_epsilon_greedy_failures
                )[RoiMetrics.ROI_SUCCESS]
                agent_stats[epsilon_key][RoiMetrics.ROI_MEAN].append(roi_mean)

                roi_means.append(roi_mean)

                steps.append(step)

            roi_mean_means.append(np.mean(roi_means))
            ax.plot(steps, roi_means)

        roi_means_mean = np.mean(roi_mean_means)
        roi_means_div = np.sqrt(np.var(roi_mean_means))
        ax.set_title(
            "$ROI_{t+1}$ of Agent: " + f"'{agent_key}'\n"
            + "$\hat{\mu}_{ROI}="
            + "{0:.5f}".format(round(roi_means_mean, 5))
            + "$, "
            + "$\hat{\sigma}_{ROI}="
            + "{0:.5f}".format(round(roi_means_div, 5))
            + "$"
        )
        ax.legend(labels, loc = 10)
        ax.set_ylabel('ROI')

        agent_roi_stats[agent_key] = agent_stats

    plt.subplots_adjust(hspace = .5)
    plt.show()
    return agent_roi_stats


def verify_agents(env, number_of_users, agents):
    stat = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }

    for agent_id in agents:
        stat['Agent'].append(agent_id)
        data = deepcopy(env).generate_logs(number_of_users, agents[agent_id])
        bandits = data[data['z'] == 'bandit']
        successes = bandits[bandits['c'] == 1].shape[0]
        failures = bandits[bandits['c'] == 0].shape[0]
        stat['0.025'].append(beta.ppf(0.025, successes + 1, failures + 1))
        stat['0.500'].append(beta.ppf(0.500, successes + 1, failures + 1))
        stat['0.975'].append(beta.ppf(0.975, successes + 1, failures + 1))

    return pd.DataFrame().from_dict(stat)


def evaluate_IPS(agent, reco_log):
    ee = []
    for u in range(max(reco_log.u)):
        t = np.array(reco_log[reco_log['u'] == u].t)
        v = np.array(reco_log[reco_log['u'] == u].v)
        a = np.array(reco_log[reco_log['u'] == u].a)
        c = np.array(reco_log[reco_log['u'] == u].c)
        z = list(reco_log[reco_log['u'] == u].z)
        ps = np.array(reco_log[reco_log['u'] == u].ps)

        jj = 0

        session = OrganicSessions()
        agent.reset()
        while True:
            if jj >= len(z):
                break
            if z[jj] == 'organic':
                session.next(DefaultContext(t[jj], u), int(v[jj]))
            else:
                prob_policy = agent.act(Observation(DefaultContext(t[jj], u), session), 0, False)[
                    'ps-a']
                ee.append(c[jj] * prob_policy[int(a[jj])] / ps[jj])
                session = OrganicSessions()
            jj += 1
    return ee


def evaluate_SNIPS(agent, reco_log):
    rewards = []
    p_ratio = []
    for u in range(max(reco_log.u)):
        t = np.array(reco_log[reco_log['u'] == u].t)
        v = np.array(reco_log[reco_log['u'] == u].v)
        a = np.array(reco_log[reco_log['u'] == u].a)
        c = np.array(reco_log[reco_log['u'] == u].c)
        z = list(reco_log[reco_log['u'] == u].z)
        ps = np.array(reco_log[reco_log['u'] == u].ps)

        jj = 0

        session = OrganicSessions()
        agent.reset()
        while True:
            if jj >= len(z):
                break
            if z[jj] == 'organic':
                session.next(DefaultContext(t[jj], u), int(v[jj]))
            else:
                prob_policy = agent.act(Observation(DefaultContext(t[jj], u), session), 0, False)[
                    'ps-a']
                rewards.append(c[jj])
                p_ratio.append(prob_policy[int(a[jj])] / ps[jj])
                session = OrganicSessions()
            jj += 1
    return rewards, p_ratio


def verify_agents_IPS(reco_log, agents):
    stat = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }

    for agent_id in agents:
        ee = evaluate_IPS(agents[agent_id], reco_log)
        mean_ee = np.mean(ee)
        se_ee = np.std(ee) / np.sqrt(len(ee))
        stat['Agent'].append(agent_id)
        stat['0.025'].append(mean_ee - 2 * se_ee)
        stat['0.500'].append(mean_ee)
        stat['0.975'].append(mean_ee + 2 * se_ee)
    return pd.DataFrame().from_dict(stat)


def verify_agents_SNIPS(reco_log, agents):
    stat = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }

    for agent_id in agents:
        rewards, p_ratio = evaluate_SNIPS(agents[agent_id], reco_log)
        ee = np.asarray(rewards) * np.asarray(p_ratio)
        mean_ee = np.sum(ee) / np.sum(p_ratio)
        se_ee = np.std(ee) / np.sqrt(len(ee))
        stat['Agent'].append(agent_id)
        stat['0.025'].append(mean_ee - 2 * se_ee)
        stat['0.500'].append(mean_ee)
        stat['0.975'].append(mean_ee + 2 * se_ee)
    return pd.DataFrame().from_dict(stat)


def evaluate_recall_at_k(agent, reco_log, k = 5):
    hits = []
    for u in range(max(reco_log.u)):
        t = np.array(reco_log[reco_log['u'] == u].t)
        v = np.array(reco_log[reco_log['u'] == u].v)
        a = np.array(reco_log[reco_log['u'] == u].a)
        c = np.array(reco_log[reco_log['u'] == u].c)
        z = list(reco_log[reco_log['u'] == u].z)
        ps = np.array(reco_log[reco_log['u'] == u].ps)

        jj = 0

        session = OrganicSessions()
        agent.reset()
        while True:
            if jj >= len(z):
                break
            if z[jj] == 'organic':
                session.next(DefaultContext(t[jj], u), int(v[jj]))
            else:
                prob_policy = agent.act(Observation(DefaultContext(t[jj], u), session), 0, False)[
                    'ps-a']
                # Does the next session exist?
                if (jj + 1) < len(z):
                    # Is the next session organic?
                    if z[jj + 1] == 'organic':
                        # Whas there no click for this bandit event?
                        if not c[jj]:
                            # Generate a top-K from the probability distribution over all actions
                            top_k = set(np.argpartition(prob_policy, -k)[-k:])
                            # Is the next seen item in the top-K?
                            if v[jj + 1] in top_k:
                                hits.append(1)
                            else:
                                hits.append(0)
                session = OrganicSessions()
            jj += 1
    return hits


def verify_agents_recall_at_k(reco_log, agents, k = 5):
    stat = {
        'Agent': [],
        '0.025': [],
        '0.500': [],
        '0.975': [],
    }

    for agent_id in agents:
        hits = evaluate_recall_at_k(agents[agent_id], reco_log, k = k)
        mean_hits = np.mean(hits)
        se_hits = np.std(hits) / np.sqrt(len(hits))
        stat['Agent'].append(agent_id)
        stat['0.025'].append(mean_hits - 2 * se_hits)
        stat['0.500'].append(mean_hits)
        stat['0.975'].append(mean_hits + 2 * se_hits)
    return pd.DataFrame().from_dict(stat)


def plot_verify_agents(result):
    fig, ax = plt.subplots()
    ax.set_title('CTR Estimate for Different Agents')
    plt.errorbar(result['Agent'],
                 result['0.500'],
                 yerr = (result['0.500'] - result['0.025'],
                         result['0.975'] - result['0.500']),
                 fmt = 'o',
                 capsize = 4)
    plt.xticks(result['Agent'], result['Agent'], rotation = 'vertical')
    return fig
