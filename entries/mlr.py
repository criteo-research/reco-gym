from recogym.agents import PyTorchMLRAgent, pytorch_mlr_args
pytorch_mlr_args['num_products'] = P
agent = build_agent_init('PyTorchMLRAgent', PyTorchMLRAgent, {**pytorch_mlr_args})