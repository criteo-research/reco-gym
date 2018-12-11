from .envs import env_0_args, env_1_args
from .envs import Observation
from .envs import Configuration
from .envs import Session
from .envs import Context, DefaultContext

from gym.envs.registration import register
from .bench_agents import test_agent

register(
    id = 'reco-gym-v0',
    entry_point = 'reco_gym.envs:RecoEnv0'
)

register(
    id = 'reco-gym-v1',
    entry_point = 'reco_gym.envs:RecoEnv1'
)
