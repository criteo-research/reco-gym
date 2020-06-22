import numpy as np
import pandas as pd
from copy import deepcopy

def share_states(data):
    return {'organic' : np.sum(data["z"]=="organic")/len(data), 
            'bandit' : np.sum(data["z"]=="bandit")/len(data),
            'sale' : np.sum(data["z"]=="sale")/len(data)}

def share_sale(data):
    return {'sale_bin' : np.sum((data["z"]=='bandit') & (data["r"]>0))/np.sum(data["z"]=='bandit'), 
            'sale_tot' : np.sum(data["r"])/np.sum(data["z"]=='bandit')}

def env_infos(env):
    env = deepcopy(env)
    return {'Gamma' : env.Gamma, 'Lambda' : env.Lambda, 'beta' : env.beta, 
            'omega' : env.omega, 'user_ps' : env.user_ps_list,
           'proba_sales' : env.proba_sales, 'proba_sales_after_scaling' : env.proba_sales_after_scaling}

def count_sales_first_session(data):
    sales_first_session = {}
    no_event=[]
    for user in data['u'].unique() :
        data_user = data.loc[data['u']==user]
        try :
            first_bandit_index = data_user[data_user['z']=="bandit"].index[0]
            first_sale_index = data_user[data_user['z']=="sale"].index[0]
            sales_first_session[user]=first_sale_index<first_bandit_index
        except :
            no_event.append(user)
            
    return sales_first_session, no_event