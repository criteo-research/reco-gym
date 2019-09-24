from recogym.agents import LogregMulticlassIpsAgent, logreg_multiclass_ips_args
agent = build_agent_init('Contextual Bandit', LogregMulticlassIpsAgent, {**logreg_multiclass_ips_args,})
