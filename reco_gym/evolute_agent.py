from copy import deepcopy
from enum import Enum


class TrainingApproach(Enum):
    ALL_DATA = 0
    SLIDING_WINDOW_ALL_DATA = 1
    ALL_EXPLORATION_DATA = 2
    SLIDING_WINDOW_EXPLORATION_DATA = 3
    MOST_VALUABLE = 4
    LAST_STEP = 5


class EvolutionCase(Enum):
    SUCCESS = 0
    FAILURE = 1
    ACTIONS = 2


def evolute_agent(
        env,
        agent,
        num_initial_train_users = 100,
        num_step_users = 1000,
        num_steps = 10,
        training_approach = TrainingApproach.ALL_DATA,
        sliding_window_samples = 10000):
    initial_agent = deepcopy(agent)

    for u in range(num_initial_train_users):
        env.reset(u)
        new_observation, reward, done, _ = env.step(None)
        while not done:
            old_observation = new_observation
            action, new_observation, reward, done, _ = env.step_offline(new_observation, reward, False)
            agent.train(old_observation, action, reward, done)

    rewards = {
        EvolutionCase.SUCCESS: [],
        EvolutionCase.FAILURE: [],
        EvolutionCase.ACTIONS: dict()
    }
    training_agent = deepcopy(agent)
    samples = 0

    for action_id in range(env.config.num_products):
        rewards[EvolutionCase.ACTIONS][action_id] = [0]

    for step in range(num_steps):
        successes = 0
        failures = 0

        for u in range(num_step_users):
            env.reset(u)
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
                    should_update_training_data = action['e-g']
                elif training_approach == TrainingApproach.SLIDING_WINDOW_EXPLORATION_DATA:
                    should_update_training_data = action['e-g'] and samples % sliding_window_samples == 0
                else:
                    assert False, f"Unknown Training Approach: {training_approach}"

                if should_update_training_data:
                    training_agent.train(old_observation, action, reward, done)

                if reward:
                    successes += 1
                    rewards[EvolutionCase.ACTIONS][action['a']][-1] += 1
                else:
                    failures += 1

        agent = training_agent
        for action_id in range(env.config.num_products):
            rewards[EvolutionCase.ACTIONS][action_id].append(0)

        if training_approach == TrainingApproach.LAST_STEP:
            training_agent = deepcopy(initial_agent)
        else:
            training_agent = deepcopy(agent)

        rewards[EvolutionCase.SUCCESS].append(successes)
        rewards[EvolutionCase.FAILURE].append(failures)
    return rewards
