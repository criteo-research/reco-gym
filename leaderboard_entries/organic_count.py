from recogym import build_agent_init
from recogym.agents import OrganicCount, organic_count_args

agent = build_agent_init('OrganicCount', OrganicCount, {**organic_count_args})
