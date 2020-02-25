from recogym import build_agent_init
from recogym.agents.organic_user_count import organic_user_count_args, OrganicUserEventCounterAgent

agent = build_agent_init('OrganicUserCount', OrganicUserEventCounterAgent, {**organic_user_count_args})
