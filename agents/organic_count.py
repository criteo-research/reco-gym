import numpy as np

organic_count_args = {
    'num_products': 10
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


class OrganicCount:
    def __init__(self, args):
        # Set all key word arguments as attributes.
        for key in args:
            setattr(self, key, args[key])

        self.co_counts = np.zeros((self.num_products, self.num_products))
        self.corr = None

    def act(self, observation, reward, done):
        """Make a recommendation"""

        self.update_lpv(observation)

        return {
            'a': self.co_counts[self.last_product_viewed, :].argmax(),
            'ps': 1.0,
        }

    def train(self, observation, action, reward, done):
        """Train the model in an online fashion"""
        if observation is not None:
            A = to_categorical([o[1] for o in observation], self.num_products)
            B = A.sum(0).reshape((self.num_products, 1))
            self.co_counts = self.co_counts + np.matmul(B, B.T)

    def update_lpv(self, observation):
        """updates the last product viewed based on the observation"""
        if observation is not None:
            self.last_product_viewed = observation[-1][-1]
