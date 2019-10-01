from recogym import build_agent_init
from recogym.agents import PyTorchMLRAgent, pytorch_mlr_args

pytorch_mlr_args['n_epochs'] = 30
pytorch_mlr_args['learning_rate'] = 0.01,
pytorch_mlr_args['ll_IPS'] = False,
pytorch_mlr_args['alpha'] = 1.0
agent = build_agent_init('PyTorchMLRAgent', PyTorchMLRAgent, {**pytorch_mlr_args})
