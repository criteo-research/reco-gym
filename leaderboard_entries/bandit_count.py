from recogym import build_agent_init
from recogym.agents import BanditCount, bandit_count_args

agent = build_agent_init('BanditCount', BanditCount, {**bandit_count_args})
