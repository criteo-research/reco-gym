from recogym import build_agent_init
from recogym.agents import LogregPolyAgent, logreg_poly_args

agent = build_agent_init('Re-weighted', LogregPolyAgent, {**logreg_poly_args, 'with_ips': True, })
