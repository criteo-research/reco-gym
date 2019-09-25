from recogym.agents import BayesianAgent, bayesian_poly_args
bayesian_poly_args['aa'] = 0.001
bayesian_poly_args['bb'] = 0.001
pytorch_banditnet_args['num_products'] = P

agent = build_agent_init('BayesianAgent', BayesianAgent, {**bayesian_poly_args})