

from .envs import env_0_args, env_1_args
from .envs import Observation
from .envs import Configuration
from .envs import Session
from .envs import Context, DefaultContext

import numpy as np
# from Keras
def to_categorical(y, num_classes=None, dtype='float32'):
  y = np.array(y, dtype='int')
  input_shape = y.shape
  if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    input_shape = tuple(input_shape[:-1])
  y = y.ravel()
  if not num_classes:
    num_classes = np.max(y) + 1
  n = y.shape[0]
  categorical = np.zeros((n, num_classes), dtype=dtype)
  categorical[np.arange(n), y] = 1
  output_shape = input_shape + (num_classes,)
  categorical = np.reshape(categorical, output_shape)
  return categorical


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

