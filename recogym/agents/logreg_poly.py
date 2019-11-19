import numpy as np
from numba import njit
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from ..envs.configuration import Configuration
from .abstract import (AbstractFeatureProvider, Model, ModelBasedAgent, ViewsFeaturesProvider)

logreg_poly_args = {
    'num_products': 10,
    'random_seed': np.random.randint(2 ** 31 - 1),

    'poly_degree': 2,

    'with_ips': False,
    # Should deltas (rewards) be used to calculate weights?
    # If delta should not be used as a IPS numerator, than `1.0' is used.
    'ips_numerator_is_delta': False,
    # Should clipping be used to calculate Inverse Propensity Score?
    'ips_with_clipping': False,
    # Clipping value that limits the value of Inverse Propensity Score.
    'ips_clipping_value': 10,

    'solver': 'lbfgs',
    'max_iter': 5000,
    'with_ps_all': False,
}


@njit(nogil = True, cache = True)
def _fast_kron_cols(data, indptr, indices, actions, num_products):
    if indptr.shape[0] - 1 == actions.shape[0]:
        col_size = data.shape[0]
        kron_data = data
        with_sliding_ix = True
    else:
        assert indptr.shape[0] - 1 == 1
        col_size = data.shape[0] * actions.shape[0]
        kron_data = np.kron(data, np.ones(num_products, dtype = np.int16))
        with_sliding_ix = False

    kron_rows = np.zeros(col_size)
    kron_cols = np.zeros(col_size)
    for ix, action in enumerate(actions):
        if with_sliding_ix:
            six = indptr[ix]
            skix = six
            eix = indptr[ix + 1]
            ekix = eix
        else:
            six = indptr[0]
            eix = indptr[1]
            delta = eix - six
            skix = ix * delta + six
            ekix = ix * delta + eix

        cols = indices[six:eix]
        kron_rows[skix:ekix] = ix
        kron_cols[skix:ekix] += cols + action * num_products

    return kron_data, kron_rows, kron_cols


class SparsePolynomialFeatures:
    def __init__(self, config: Configuration):
        self.config = config

    def transform(
            self,
            features: sparse.csr_matrix,
            actions: np.ndarray
    ) -> sparse.csr_matrix:
        kron_data, kron_rows, kron_cols = _fast_kron_cols(
            features.data,
            features.indptr,
            features.indices,
            actions,
            self.config.num_products
        )
        assert kron_data.shape[0] == kron_rows.shape[0]
        assert kron_data.shape[0] == kron_cols.shape[0]
        kron = sparse.coo_matrix(
            (
                kron_data,
                (kron_rows, kron_cols)
            ),
            (actions.shape[0], self.config.num_products * self.config.num_products)
        )

        if features.shape[0] != actions.shape[0]:
            features_data = np.tile(features.data, actions.shape[0])
            features_cols = np.tile(features.indices, actions.shape[0])
            feature_rows = np.repeat(np.arange(actions.shape[0]), features.nnz)
            features = sparse.coo_matrix(
                (
                    features_data,
                    (feature_rows, features_cols)
                ),
                (actions.shape[0], self.config.num_products),
                dtype = np.int16
            )

        actions = sparse.coo_matrix(
            (
                actions,
                (range(actions.shape[0]), actions)
            ),
            (actions.shape[0], self.config.num_products),
            dtype = np.int16
        )

        assert features.shape[0] == actions.shape[0]
        assert features.shape[0] == kron.shape[0]
        assert features.shape[1] == self.config.num_products
        assert actions.shape[1] == self.config.num_products

        return sparse.hstack(
            (features, actions, kron)
        )


class LogregPolyModelBuilder(AbstractFeatureProvider):
    def __init__(self, config, is_sparse=False):
        super(LogregPolyModelBuilder, self).__init__(config, is_sparse=is_sparse)

    def build(self):
        class LogisticRegressionPolyFeaturesProvider(ViewsFeaturesProvider):
            """
            Logistic Regression Polynomial Feature Provider.
            """

            def __init__(self, config, poly):
                super(LogisticRegressionPolyFeaturesProvider, self).__init__(config, is_sparse=True)
                self.poly = poly
                self.all_actions = np.arange(self.config.num_products)

            def features(self, observation):
                return self.poly.transform(
                    super().features(observation).tocsr(),
                    self.all_actions
                )

        class LogisticRegressionModel(Model):
            """
            Logistic Regression Model
            """

            def __init__(self, config, logreg):
                super(LogisticRegressionModel, self).__init__(config)
                self.logreg = logreg

            def act(self, observation, features):
                action_proba = self.logreg.predict_proba(features)[:, 1]
                action = np.argmax(action_proba)
                if self.config.with_ps_all:
                    ps_all = np.zeros(self.config.num_products)
                    ps_all[action] = 1.0
                else:
                    ps_all = ()
                return {
                    **super().act(observation, features),
                    **{
                        'a': action,
                        'ps': 1.0,
                        'ps-a': ps_all,
                    },
                }

        features, actions, deltas, pss = self.train_data()

        logreg = LogisticRegression(
            solver = self.config.solver,
            max_iter = self.config.max_iter,
            random_state = self.config.random_seed,
            n_jobs = -1
        )

        poly = SparsePolynomialFeatures(self.config)
        features_poly = poly.transform(features, actions)

        if self.config.with_ips:
            ips_numerator = deltas if self.config.ips_numerator_is_delta else 1.0
            weights = ips_numerator / pss
            if self.config.ips_with_clipping:
                weights = np.minimum(deltas / pss, self.config.ips_clipping_value)
            lr = logreg.fit(features_poly, deltas, weights)
        else:
            lr = logreg.fit(features_poly, deltas)
        print('Model was built!')

        return (
            LogisticRegressionPolyFeaturesProvider(self.config, poly),
            LogisticRegressionModel(self.config, lr)
        )


class LogregPolyAgent(ModelBasedAgent):
    """
    Logistic Regression Polynomial Agent
    """

    def __init__(self, config = Configuration(logreg_poly_args)):
        super(LogregPolyAgent, self).__init__(
            config,
            LogregPolyModelBuilder(config, is_sparse=True)
        )
