class TimeGenerator:
    """"""
    def __init__(self, config):
        self.config = config

    def new_time(self):
        raise NotImplemented

    def reset(self):
        raise NotImplemented
