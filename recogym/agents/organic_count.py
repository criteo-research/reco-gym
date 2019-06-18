import numpy as np

from recogym.agents import Agent
from recogym import Configuration

organic_count_args = {
    'num_products': 10,
}


# From Keras.
def to_categorical(y, num_classes = None, dtype = 'float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype = 'int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype = dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class OrganicCount(Agent):
    """
    Organic Count

    The Agent that selects an Action based on the most frequently viewed Product.
    """

    def __init__(self, config = Configuration(organic_count_args)):
        super(OrganicCount, self).__init__(config)

        self.co_counts = np.zeros((self.config.num_products, self.config.num_products))
        self.corr = None

    def act(self, observation, reward, done):
        """Make a recommendation"""

        self.update_lpv(observation)

        action = self.co_counts[self.last_product_viewed, :].argmax()
        ps_all = np.zeros(self.config.num_products)
        ps_all[action] = 1.0
        return {
            **super().act(observation, reward, done),
            **{
                'a': self.co_counts[self.last_product_viewed, :].argmax(),
                'ps': 1.0,
                'ps-a': ps_all,
            },
        }

    def train(self, observation, action, reward, done = False):
        """Train the model in an online fashion"""
        if observation.sessions():
            A = to_categorical(
                [session['v'] for session in observation.sessions()],
                self.config.num_products
            )
            B = A.sum(0).reshape((self.config.num_products, 1))
            self.co_counts = self.co_counts + np.matmul(B, B.T)

    def update_lpv(self, observation):
        """updates the last product viewed based on the observation"""
        if observation.sessions():
            self.last_product_viewed = observation.sessions()[-1]['v']
