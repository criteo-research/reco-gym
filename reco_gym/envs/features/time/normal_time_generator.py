from numpy.random.mtrand import RandomState
import numpy as np

from .time_generator import TimeGenerator


class NormalTimeGenerator(TimeGenerator):
    """"""
    def __init__(self, config):
        super(NormalTimeGenerator, self).__init__(config)
        self.current_time = 0

        if not hasattr(self.config, 'normal_time_mu'):
            self.normal_time_mu = 0
        else:
            self.normal_time_mu = self.config.normal_time_mu

        if not hasattr(self.config, 'normal_time_sigma'):
            self.normal_time_sigma = 1
        else:
            self.normal_time_sigma = self.config.normal_time_sigma

        self.rng = RandomState(config.random_seed)

    def new_time(self):
        tmp_time = self.current_time
        self.current_time += np.abs(self.rng.normal(self.normal_time_mu, self.normal_time_sigma))
        return tmp_time

    def reset(self):
        self.current_time = 0
