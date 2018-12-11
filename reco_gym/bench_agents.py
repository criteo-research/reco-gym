from scipy.stats.distributions import beta


def test_agent(
        env,
        agent,
        num_offline_users = 1000,
        num_online_users = 100,
        num_organic_offline_users = 100,
        num_epochs = 1
):
    # Offline organic Training -------------------------------------------------------
    print("Starting Agent Training")
    for i in range(num_epochs):
        for u in range(num_organic_offline_users):
            env.reset(u)
            observation, _, _, _ = env.step(None)
            agent.train(observation, None, None, True)

    # Offline Training -------------------------------------------------------
    for i in range(num_epochs):
        for u in range(num_offline_users):
            env.reset(u)
            observation, _, done, _ = env.step(None)
            while not done:
                old_observation = observation
                action, observation, reward, done, info = env.step_offline(old_observation, 0, False)
                agent.train(old_observation, action, reward, done)

    # Online Testing ---------------------------------------------------------
    suc = 0
    fail = 0
    print("Starting Agent Testing")
    for u in range(num_online_users):
        env.reset(u)
        observation, _, done, _ = env.step(None)
        reward = None
        done = None
        while not done:
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action['a'])

            if reward:
                suc += 1
            else:
                fail += 1

    return (
        beta.ppf(0.5, suc + 1, fail + 1),
        beta.ppf(0.025, suc + 1, fail + 1),
        beta.ppf(0.975, suc + 1, fail + 1)
    )
