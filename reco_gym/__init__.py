from .envs import env_0_args, env_1_args
from .envs import Observation
from .envs import Configuration
from .envs import Session
from .envs import Context, DefaultContext

from .constants import (
    AgentStats,
    AgentInit,
    EvolutionCase,
    TrainingApproach,
)
from .bench_agents import test_agent
from .evolute_agent import evolute_agent
from .evolute_agent import (
    build_agent_init,
    gather_agent_stats,
    plot_agent_stats
)

from .envs.features.time.default_time_generator import DefaultTimeGenerator
from .envs.features.time.normal_time_generator import NormalTimeGenerator

from gym.envs.registration import register

register(
    id = 'reco-gym-v0',
    entry_point = 'reco_gym.envs:RecoEnv0'
)

register(
    id = 'reco-gym-v1',
    entry_point = 'reco_gym.envs:RecoEnv1'
)
