import numpy as np
from scipy.sparse import vstack, csr_matrix
import gym, recogym
import pandas as pd
import math
from copy import deepcopy
from tqdm import tqdm, trange
from recogym import env_1_args, Configuration
from recogym.agents import RandomAgent, random_args
import sys
from multiprocessing import Pool

def gen_data(data, num_products, va_ratio=0.2, te_ratio=0.2):
    data = pd.DataFrame().from_dict(data)

    global process_helper
    def process_helper(user_id):
        tmp_feature = []
        tmp_action = []
        tmp_ps = []
        tmp_delta = []
        tmp_set_flag = []

        views = np.zeros((0, num_products))
        history = np.zeros((0, 1))
        for _, user_datum in data[data['u'] == user_id].iterrows():
            assert (not math.isnan(user_datum['t']))
            if user_datum['z'] == 'organic':
                assert (math.isnan(user_datum['a']))
                assert (math.isnan(user_datum['c']))
                assert (not math.isnan(user_datum['v']))

                view = int(user_datum['v'])

                tmp_view = np.zeros(num_products)
                tmp_view[view] = 1

                # Append the latest view at the beginning of all views.
                views = np.append(tmp_view[np.newaxis, :], views, axis = 0)
                history = np.append(np.array([user_datum['t']])[np.newaxis, :], history, axis = 0)
            else:
                assert (user_datum['z'] == 'bandit')
                assert (not math.isnan(user_datum['a']))
                assert (not math.isnan(user_datum['c']))
                assert (math.isnan(user_datum['v']))

                action = int(user_datum['a'])
                delta = int(user_datum['c'])
                ps = user_datum['ctr']
                time = user_datum['t']

                train_views = views

                feature = np.sum(train_views, axis = 0)
                feature = feature/np.linalg.norm(feature)

                tmp_feature.append(feature)
                tmp_action.append(action)
                tmp_delta.append(delta)
                tmp_ps.append(ps)
                tmp_set_flag.append(-1) # user without enough bandits will be removed

        tmp_set_flag = np.array(tmp_set_flag)
        va_num = math.ceil(va_ratio*tmp_set_flag.shape[0]) 
        te_num = math.ceil(te_ratio*tmp_set_flag.shape[0]) 
        if va_num + te_num < tmp_set_flag.shape[0]:
            tmp_set_flag[:-1*(va_num+te_num)] = 0
            tmp_set_flag[-1*(va_num+te_num):] = 1
            tmp_set_flag[-1*te_num:] = 2

        return csr_matrix(np.array(tmp_feature)), \
                np.array(tmp_action), \
                np.array(tmp_delta), \
                np.array(tmp_ps), \
                np.array(tmp_set_flag)

    with Pool(8) as p:
        output = p.map(process_helper, data['u'].unique())
        features, actions, deltas, pss, set_flags = zip(*output)

    return features, actions, deltas, pss, set_flags

def dump_svm(f, X, y_idx, y_propensity, y_value):
    if not hasattr(f, "write"):
        f = open(f, "w")
    X_is_sp = int(hasattr(X, "tocsr"))
    #y_is_sp = int(hasattr(y, "tocsr"))

    value_pattern = "%d:%.6g"
    label_pattern = "%d:%d:%.16g"
    line_pattern = "%s %s\n"
    
    for i, d in enumerate(zip(y_idx, y_value, y_propensity)):
        if X_is_sp:
            span = slice(X.indptr[i], X.indptr[i + 1])
            row = zip(X.indices[span], X.data[span])
        else:
            nz = X[i] != 0
            row = zip(np.where(nz)[0], X[i, nz])

        s = " ".join(value_pattern % (j, x) for j, x in row)
        labels_str = label_pattern % d 
        feat = (labels_str, s)
        f.write(line_pattern % feat)
    
    return

def main():
    print('Need to change rng of user embedding generation first')
    root = sys.argv[1]
    P = 100
    U = 20000
    
    env_1_args['random_seed'] = 8964
    env_1_args['random_seed_for_user'] = 2
    env_1_args['num_products'] = P
    env_1_args['K'] = 5
    env_1_args['sigma_omega'] = 0  # default 0.1, the varaince of user embedding changes with time.
    env_1_args['number_of_flips'] = P//2
    env_1_args['prob_leave_bandit'] = float(sys.argv[2])
    env_1_args['prob_leave_organic'] = 0.0
    env_1_args['prob_bandit_to_organic'] = 1 - env_1_args['prob_leave_bandit']
    env_1_args['prob_organic_to_bandit'] = 0.1
    
    
    env = gym.make('reco-gym-v1')
    env.init_gym(env_1_args)
    
    data = env.generate_gt(U)
    data.to_csv('%s/data_%d_%d.csv'%(root, P, U), index=False)
    #data = pd.read_csv('%s/data_%d_%d.csv'%(root, P, U))
    
    features, actions, deltas, pss, set_flags = gen_data(data, P)
    tr_num = int(U*0.)
    va_num = int(U*0.)
    with open('%s/te.nonmerge.uniform.svm'%root, 'w') as te:
        dump_svm(te, vstack(features[(tr_num+va_num):]), \
                np.hstack(actions[(tr_num+va_num):]), \
                np.hstack(pss[(tr_num+va_num):]), \
                np.hstack(deltas[(tr_num+va_num):]))
    
    with open('%s/label.svm'%root, 'w') as label:
        for i in range(P):
            label.write('%d:1\n'%i)

if __name__ == '__main__':
    main()
