from recogym import build_agent_init
from recogym.agents import BayesianAgentVB, bayesian_poly_args
bayesian_poly_args['aa'] = 0.1
bayesian_poly_args['bb'] = 0.1
bayesian_poly_args['num_products'] = P

agent = build_agent_init('BayesianAgentVB', BayesianAgentVB, {**bayesian_poly_args})