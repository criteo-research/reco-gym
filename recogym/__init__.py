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
    RoiMetrics
)
from .bench_agents import test_agent

from .evaluate_agent import (
    evaluate_agent,
    build_agent_init,
    gather_agent_stats,
    plot_agent_stats,
    gather_exploration_stats,
    plot_evolution_stats,
    plot_heat_actions,
    plot_roi,
    verify_agents,
    verify_agents_IPS,
    to_categorical
)

from .competition import competition_score

from .envs.features.time.default_time_generator import DefaultTimeGenerator
from .envs.features.time.normal_time_generator import NormalTimeGenerator

from gym.envs.registration import register

register(
    id = 'reco-gym-v0',
    entry_point = 'recogym.envs:RecoEnv0'
)

register(
    id = 'reco-gym-v1',
    entry_point = 'recogym.envs:RecoEnv1'
)

