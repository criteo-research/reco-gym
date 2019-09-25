from recogym.agents import PyTorchBanditNetAgent, pytorch_banditnet_args
pytorch_banditnet_args['num_products'] = P
agent = build_agent_init('PyTorchBanditNetAgent', PyTorchBanditNetAgent, {**pytorch_banditnet_args})