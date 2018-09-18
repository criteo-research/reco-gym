from scipy.stats.distributions import beta


class LogFile:
    """simple helper class for logging"""
    def __init__(self, path):

        if path is not None:
            self.f = open(path, 'w')
            self.f.write('user_id|is_online|observation|action|reward\n')
        else:
            self.f = None

    def close(self):
        if self.f is not None:
            self.f.close()

    def write(self, user_id, is_online, observation, action, reward):
        if self.f is not None:
            line = '|'.join([str(user_id), str(is_online), str(observation),
                             str(action), str(reward)])
            self.f.write(line + '\n')


def test_agent(env, agent, num_offline_users=1000, num_online_users=100,num_organic_offline_users=100,
               num_epochs=1, log_file=None):

    # open optional logging
    log = LogFile(log_file)

    # initialize user id to 1 for logging purposes
    user_id = 1

    # Offline organic Training -------------------------------------------------------
    print("Starting Agent Training")
    for i in range(num_epochs):
        env.__init__()  # Reset the env for repeated sequences
        for u in range(num_organic_offline_users):
            env.reset()
            observation, _, _, _ = env.step(None)
            agent.train(observation, None, None, True)


    # Offline Training -------------------------------------------------------
    for i in range(num_epochs):
        env.__init__()  # Reset the env for repeated sequences
        for u in range(num_offline_users):
            env.reset()
            observation, _, done, _ = env.step(None)
            while not done:
                old_observation = observation
                action, observation, reward, done, info = env.step_offline()
                agent.train(old_observation, action, reward, done)
                if i == (num_epochs-1):
                    log.write(user_id, False, observation, action, reward)
            if i == (num_epochs-1):
                user_id += 1

    # Online Testing ---------------------------------------------------------
    suc = 0
    fail = 0
    print("Starting Agent Testing")
    for _ in range(num_online_users):
        env.reset()
        observation, _, done, _ = env.step(None)
        reward = None
        done = None
        while not done:
            action = agent.act(observation, reward, done)
            observation, reward, done, info = env.step(action)

            user_id += 1
            if reward:
                suc = suc + 1
            else:
                fail = fail + 1

    return (
        beta.ppf(0.5, suc+1, fail+1),
        beta.ppf(0.025, suc+1, fail+1),
        beta.ppf(0.975, suc+1, fail+1)
        )
