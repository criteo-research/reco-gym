from recogym import build_agent_init
from recogym.agents import LogregPolyAgent, logreg_poly_args
logreg_poly_args['num_products'] = P
agent = build_agent_init('Re-weighted', LogregPolyAgent, {**logreg_poly_args, 'with_ips': True,})