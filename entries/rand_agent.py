from recogym import build_agent_init
from recogym.agents import RandomAgent, random_args
random_args['num_products'] = P
agent = build_agent_init('RandomAgent', RandomAgent, {**random_args})