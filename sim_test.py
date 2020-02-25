import argparse
import datetime
import glob
import os
import types

import pandas as pd

from recogym import (
    competition_score,
    AgentInit,
)

if __name__ == "__main__":
    import tensorflow as tf2
    print(f'TensorFlow V2: {tf2.__version__}')
    import tensorflow.compat.v1 as tf1
    print(f'TensorFlow V2: {tf1.__version__}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--P', type=int, default=100, help='Number of products')
    parser.add_argument('--UO', type=int, default=100, help='Number of organic users to train on')
    parser.add_argument('--U', type=int, default=100, help='Number of users to train on')
    parser.add_argument('--Utest', type=int, default=1000, help='Number of users to test')
    parser.add_argument('--seed', type=int, default=100, help='Seed')
    parser.add_argument('--K', type=int, default=20, help='Number of latent factors')
    parser.add_argument('--F', type=int, default=20,
                        help='Number of flips, how different is bandit from organic')
    parser.add_argument('--log_epsilon', type=float, default=0.05,
                        help='Pop logging policy epsilon')
    parser.add_argument('--sigma_omega', type=float, default=0.01, help='sigma_omega')
    parser.add_argument('--entries_dir', type=str, default='my_entries',
                        help='directory with agent files for a leaderboard of small baselines for P small try setting to leaderboard_entries')
    parser.add_argument('--with_cache', type=bool, default=False,
                        help='Do use cache for training data or not')

    args = parser.parse_args()

    P, UO, U, Utest, seed, num_flips, K, sigma_omega, log_epsilon, entries_dir, with_cache = (
        args.P,
        args.UO,
        args.U,
        args.Utest,
        args.seed,
        args.F,
        args.K,
        args.sigma_omega,
        args.log_epsilon,
        args.entries_dir,
        args.with_cache,
    )

    print(args)

    adf = []
    start = datetime.datetime.now()
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    for agent_file in glob.glob(entries_dir + '/*.py'):
        print(f'Agent: {agent_file}')
        try:
            tmp_module = types.ModuleType('tmp_module')
            exec(
                open(agent_file).read(),
                tmp_module.__dict__
            )
            if hasattr(tmp_module, 'TestAgent'):
                agent_class = tmp_module.TestAgent
                agent_configs = tmp_module.test_agent_args
                agent_name = 'Test Agent'
            else:
                if hasattr(tmp_module, 'agent'):
                    for agent_key in tmp_module.agent.keys():
                        agent_class = tmp_module.agent[agent_key][AgentInit.CTOR]
                        agent_configs = tmp_module.agent[agent_key][AgentInit.DEF_ARGS]
                        agent_name = agent_key
                else:
                    print('There is no Agent to test!')
                    continue

            df = competition_score(
                P,
                UO,
                U,
                Utest,
                seed,
                K,
                num_flips,
                log_epsilon,
                sigma_omega,
                agent_class,
                agent_configs,
                agent_name,
                with_cache
            )

            df = df.join(pd.DataFrame({
                'entry': [agent_file]
            }))

            print(df)

            adf.append(df)
        except Exception as ex:
            print(f'Agent @ "{agent_file}" failed: {str(ex)}')

    out_dir = entries_dir + '_' + str(P) + '_' + str(U) + '_' + str(Utest) + '_' + str(start)
    os.mkdir(out_dir)
    fp = open(out_dir + '/config.txt', 'w')
    fp.write(str(args))
    fp.close()

    leaderboard = pd.concat(adf)
    leaderboard = leaderboard.sort_values(by='q0.500', ascending=False)
    leaderboard.to_csv(out_dir + '/leaderboard.csv')
