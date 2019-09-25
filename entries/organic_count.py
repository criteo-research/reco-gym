from recogym.agents import OrganicCount, organic_count_args
organic_count_args['num_products'] = P
agent = build_agent_init('OrganicCount', OrganicCount, {**organic_count_args})