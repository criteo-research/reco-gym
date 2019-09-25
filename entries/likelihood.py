from recogym.agents import LogregPolyAgent, logreg_poly_args
logreg_poly_args['num_products'] = P
agent = build_agent_init('likelihood', LogregPolyAgent, {**logreg_poly_args,})

