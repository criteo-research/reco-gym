class Configuration:
    """
    Configuration

    That class defines Environment Configurations used in RecoGym.
    The configurations are provided as a dictionary: key = value.
    The value can be ANY type i.e. a complex object, function etc.

    The class is immutable i.e. once an instance of that class is created,
    no configuration can be changed.
    """

    def __init__(self, args):
        # self.args = args
        # Set all key word arguments as attributes.
        for key in args:
            super(Configuration, self).__setattr__(key, args[key])
        Configuration.__slots__ = [key for key in args]

    def __setattr__(self, key, value):
        pass

    def __deepcopy__(self, memodict={}):
        return self
