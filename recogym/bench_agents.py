import datetime
import hashlib
import json
import os
import pickle
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
from scipy.stats.distributions import beta
from tqdm import trange, tqdm

from recogym import AgentStats, DefaultContext, Observation
from recogym.envs.session import OrganicSessions

CACHE_DIR = os.path.join(os.path.join(str(Path.home()), '.reco-gym'), 'cache')


def _cache_file_name(env, num_organic_offline_users: int, num_offline_users: int) -> str:
    unique_config_data = (
        (
            env.config.K,
            (
                str(type(env.config.agent)),
            ),
            env.config.change_omega_for_bandits,
            env.config.normalize_beta,
            env.config.num_clusters,
            env.config.num_products,
            env.config.num_users,
            env.config.number_of_flips,
            env.config.phi_var,
            env.config.prob_bandit_to_organic,
            env.config.prob_leave_bandit,
            env.config.prob_leave_organic,
            env.config.prob_organic_to_bandit,
            env.config.random_seed,
            env.config.sigma_mu_organic,
            env.config.sigma_omega,
            env.config.sigma_omega_initial,
            env.config.with_ps_all,
        ),
        num_organic_offline_users,
        num_offline_users
    )
    return f'{hashlib.sha1(json.dumps(unique_config_data).encode()).hexdigest()}.pkl'


def _cached_data(env, num_organic_offline_users: int, num_offline_users: int) -> str:
    cache_file_name = _cache_file_name(env, num_organic_offline_users, num_offline_users)
    file_path = os.path.join(CACHE_DIR, cache_file_name)
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file, fix_imports=False)
    else:
        data = env.generate_logs(num_offline_users=num_offline_users,
                                 num_organic_offline_users=num_organic_offline_users)
        with open(file_path, 'wb') as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL, fix_imports=False)
    return data


def _collect_stats(args):
    env = args['env']
    agent = args['agent']
    num_offline_users = args['num_offline_users']
    num_online_users = args['num_online_users']
    num_organic_offline_users = args['num_organic_offline_users']
    epoch_with_random_reset = args['epoch_with_random_reset']
    epoch = args['epoch']
    with_cache = args['with_cache']

    print(f"START: Agent Training #{epoch}")

    unique_user_id = 0
    new_agent = deepcopy(agent)

    print(f"START: Agent Training @ Epoch #{epoch}")
    start = time.time()

    if epoch_with_random_reset:
        train_env = deepcopy(env)
        train_env.reset_random_seed(epoch)
    else:
        train_env = env

    if with_cache:
        data = _cached_data(train_env, num_organic_offline_users, num_offline_users)

        def _train(observation, session, action, reward, time, done):
            if observation:
                assert session is not None
            else:
                observation = Observation(DefaultContext(time, current_user), session)
            new_agent.train(observation, action, reward, done)
            return None, OrganicSessions(), None, None

        current_session = OrganicSessions()
        last_observation = None
        last_action = None
        last_reward = None
        last_time = None

        current_user = None
        with tqdm(total=data.shape[0], desc='Offline Logs') as pbar:
            for _, row in data.iterrows():
                pbar.update()
                t, u, z, v, a, c, ps, ps_a = row.values

                if current_user is None:
                    current_user = u

                if current_user != u:
                    last_observation, current_session, last_action, last_reward = _train(
                        last_observation,
                        current_session,
                        last_action,
                        last_reward,
                        last_time,
                        True
                    )
                    current_user = u

                if last_action:
                    last_observation, current_session, last_action, last_reward = _train(
                        last_observation,
                        current_session,
                        last_action,
                        last_reward,
                        last_time,
                        False
                    )

                if z == 'organic':
                    assert (not np.isnan(v))
                    assert (np.isnan(a))
                    assert (np.isnan(c))
                    current_session.next(DefaultContext(t, u), np.int16(v))
                else:
                    last_observation = Observation(DefaultContext(t, u), current_session)
                    current_session = OrganicSessions()
                    assert (np.isnan(v))
                    assert (not np.isnan(a))
                    last_action = {
                        't': t,
                        'u': u,
                        'a': np.int16(a),
                        'ps': ps,
                        'ps-a': ps_a,
                    }
                    assert (not np.isnan(c))
                    last_reward = c

                last_time = t

            _train(
                last_observation,
                current_session,
                last_action,
                last_reward,
                last_time,
                True
            )
    else:
        # Offline Organic Training.
        for _ in trange(num_organic_offline_users, desc='Organic Users'):
            train_env.reset(unique_user_id)
            unique_user_id += 1
            new_observation, _, _, _ = train_env.step(None)
            new_agent.train(new_observation, None, None, True)

        # Offline Organic and Bandit Training.
        for _ in trange(num_offline_users, desc='Users'):
            train_env.reset(unique_user_id)
            unique_user_id += 1
            new_observation, _, done, reward = train_env.step(None)
            while not done:
                old_observation = new_observation
                action, new_observation, reward, done, _ = train_env.step_offline(
                    old_observation, reward, done
                )
                new_agent.train(old_observation, action, reward, False)
            old_observation = new_observation
            action, _, reward, done, _ = train_env.step_offline(
                old_observation, reward, done
            )
            new_agent.train(old_observation, action, reward, True)
    print(f"END: Agent Training @ Epoch #{epoch} ({time.time() - start}s)")

    # Online Testing.
    print(f"START: Agent Evaluating @ Epoch #{epoch}")
    start = time.time()

    if epoch_with_random_reset:
        eval_env = deepcopy(env)
        eval_env.reset_random_seed(epoch)
    else:
        eval_env = env

    stat_data = eval_env.generate_logs(num_offline_users=num_online_users, agent=new_agent)
    rewards = stat_data[~np.isnan(stat_data['a'])]['c']
    successes = np.sum(rewards)
    failures = rewards.shape[0] - successes
    print(f"END: Agent Evaluating @ Epoch #{epoch} ({time.time() - start}s)")

    return {
        AgentStats.SUCCESSES: successes,
        AgentStats.FAILURES: failures,
    }


def test_agent(
        env,
        agent,
        num_offline_users=1000,
        num_online_users=100,
        num_organic_offline_users=100,
        num_epochs=1,
        epoch_with_random_reset=False,
        with_cache=False
):
    successes = 0
    failures = 0

    argss = [
        {
            'env': env,
            'agent': agent,
            'num_offline_users': num_offline_users,
            'num_online_users': num_online_users,
            'num_organic_offline_users': num_organic_offline_users,
            'epoch_with_random_reset': epoch_with_random_reset,
            'epoch': epoch,
            'with_cache': with_cache,
        }
        for epoch in range(num_epochs)
    ]

    for result in [_collect_stats(args) for args in argss]:
        successes += result[AgentStats.SUCCESSES]
        failures += result[AgentStats.FAILURES]

    return (
        beta.ppf(0.500, successes + 1, failures + 1),
        beta.ppf(0.025, successes + 1, failures + 1),
        beta.ppf(0.975, successes + 1, failures + 1)
    )
