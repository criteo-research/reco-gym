from recogym import build_agent_init
from recogym.agents import PyTorchMLRAgent, pytorch_mlr_args
pytorch_mlr_args['n_epochs'] = 30
pytorch_mlr_args['learning_rate'] = 0.01

agent = build_agent_init('PyTorchMLRAgent', PyTorchMLRAgent, {**pytorch_mlr_args})
