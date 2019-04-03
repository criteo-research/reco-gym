from .time_generator import TimeGenerator


class DefaultTimeGenerator(TimeGenerator):
    """"""
    def __init__(self, config):
        super(DefaultTimeGenerator, self).__init__(config)
        self.current_time = 0

    def new_time(self):
        tmp_time = self.current_time
        self.current_time += 1
        return tmp_time

    def reset(self):
        self.current_time = 0
