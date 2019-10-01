import multiprocessing
import time
from copy import deepcopy
from multiprocessing import Pool

from scipy.stats.distributions import beta

from recogym import AgentStats


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
            action, new_observation, reward, done, info = (
                env.step_offline(old_observation, 0, False)
            )
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

    for result in [_collect_stats(args) for args in argss]:
        successes += result[AgentStats.SUCCESSES]
        failures += result[AgentStats.FAILURES]

    return (
        beta.ppf(0.500, successes + 1, failures + 1),
        beta.ppf(0.025, successes + 1, failures + 1),
        beta.ppf(0.975, successes + 1, failures + 1)
    )
