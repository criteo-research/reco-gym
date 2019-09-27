from recogym import build_agent_init
from recogym.agents import BanditCount, bandit_count_args
bandit_count_args['num_products'] = P
agent = build_agent_init('BanditCount', BanditCount, {**bandit_count_args})