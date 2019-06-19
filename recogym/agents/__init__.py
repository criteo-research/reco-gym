import warnings

from .abstract import (
    Agent,
    FeatureProvider,
    AbstractFeatureProvider,
    ViewsFeaturesProvider,
    Model,
    ModelBasedAgent
)
from .bandit_count import BanditCount, bandit_count_args
from .random_agent import RandomAgent, random_args
from .organic_count import OrganicCount, organic_count_args
from .logreg_ips import LogregMulticlassIpsAgent, logreg_multiclass_ips_args
from .epsilon_greedy import EpsilonGreedy, epsilon_greedy_args
from .logreg_poly import LogregPolyAgent, logreg_poly_args
from .organic_user_count import OrganicUserEventCounterAgent, organic_user_count_args

# These agents require pytorch to work correctly
try:
    import torch

    import_torch_agents = True
except ImportError as e:
    import_torch_agents = False

if import_torch_agents:
    from .bandit_mf import BanditMFSquare, bandit_mf_square_args
    from .nn_ips import NnIpsAgent, nn_ips_args
    from .organic_mf import OrganicMFSquare, organic_mf_square_args
else:
    warnings.warn('Agents Bandit MF Square, Organic MF Square and NN IPS are not available '
                  'since torch cannot be imported. '
                  'Install it with `pip install torch` and test it with `python -c "import torch"`')
