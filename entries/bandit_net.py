from recogym.agents import PyTorchBanditNetAgent, pytorch_banditnet_args
agent = build_agent_init('PyTorchBanditNetAgent', PyTorchBanditNetAgent, {**pytorch_banditnet_args})