from recogym.agents import BanditMFSquare, bandit_mf_square_args
bandit_mf_square_args['num_products'] = P
agent = build_agent_init('BanditMFsquare', BanditMFSquare, {**bandit_mf_square_args})