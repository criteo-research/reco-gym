from recogym.agents import PyTorchMLRAgent, pytorch_mlr_args
agent = build_agent_init('PyTorchMLRAgent', PyTorchMLRAgent, {**pytorch_mlr_args})