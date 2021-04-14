'''
Copyright (c) 2020, AGH University of Science and Technology.
'''
import unittest

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors

from keras_svi import svi


class TestSVI(unittest.TestCase):
    def test_svi(self):
        nn = 50
        _x = np.random.normal(size=(nn, 3)).astype(np.float32)
        _y = .4 * _x[:, 0] + 0.2 * _x[:, 1] - 0.3 * _x[:, 2] + 1.5 + np.random.normal(size=(nn), scale=.1).astype(
            np.float32)

        _x = tf.convert_to_tensor(_x)
        _y = tf.convert_to_tensor(_y)

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(3)),
            tf.keras.layers.Dense(1)
        ])

        bayesian_model = svi.SVI(model, kl_scale=1.0)
        bayesian_model.compile(loss=tfk.losses.MeanSquaredError(),
                               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
        bayesian_model.fit(_x, _y, epochs=2, batch_size=nn, verbose=1)

        return

    def test_built(self):
        nn = 50
        _x = np.random.normal(size=(nn, 3)).astype(np.float32)
        _y = .4 * _x[:, 0] + 0.2 * _x[:, 1] - 0.3 * _x[:, 2] + 1.5 + np.random.normal(size=(nn), scale=.1).astype(
            np.float32)

        _x = tf.convert_to_tensor(_x)
        _y = tf.convert_to_tensor(_y)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])
        model(_x)
        bayesian_model = svi.SVI(model, kl_scale=1.0)
        bayesian_model.compile(loss=tfk.losses.MeanSquaredError(),
                               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
        bayesian_model.fit(_x, _y, epochs=2, batch_size=nn, verbose=1)

        return

    def test_rnn(self):
        x = np.random.normal(size=(200, 10, 1))
        y = np.random.normal(size=(200, 1))
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(10, 1)),
            # tf.keras.layers.GRU(4),
            tf.keras.layers.RNN(tfk.layers.GRUCell(4)),
            tf.keras.layers.Dense(1)
        ])

        bayesian_model = svi.SVI(model, kl_scale=1.0)
        bayesian_model.compile(loss=tfk.losses.MeanSquaredError(),
                               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
        bayesian_model.fit(x, y, epochs=2, batch_size=64, verbose=1)

    def test_map(self):
        nn = 50
        _x = np.random.normal(size=(nn, 3))
        _y = .4 * _x[:, 0] + 0.2 * _x[:, 1] - 0.3 * _x[:, 2] + 1.5 + np.random.normal(size=(nn), scale=.1)

        _x = tf.convert_to_tensor(_x)
        _y = tf.convert_to_tensor(_y)

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1)
        ])

        def _make_posterior(v):
            n = len(v.shape)
            return tfd.Independent(tfd.Deterministic(tf.Variable(tf.convert_to_tensor(v))),
                                   reinterpreted_batch_ndims=n)

        def _make_prior(posterior):
            n = len(posterior.event_shape)
            return tfd.Independent(tfd.Normal(tf.zeros(posterior.event_shape, dtype=posterior.dtype),
                                              tf.constant(2., dtype=posterior.dtype)),
                                   reinterpreted_batch_ndims=n)

        bayesian_model = svi.SVI(model,
                                 kl_scale=1.0,
                                 prior_fn=_make_prior,
                                 posterior_fn=_make_posterior
                                 )

        bayesian_model.compile(
            loss=tfk.losses.MeanSquaredError(),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
            run_eagerly=False
        )
        bayesian_model.fit(_x, _y, epochs=2, batch_size=nn, verbose=1)

        return

    def test_mc_kl(self):
        nn = 50
        _x = np.random.normal(size=(nn, 3))
        _y = .4 * _x[:, 0] + 0.2 * _x[:, 1] - 0.3 * _x[:, 2] + 1.5 + np.random.normal(size=(nn), scale=.1)

        _x = tf.convert_to_tensor(_x)
        _y = tf.convert_to_tensor(_y)

        model = tf.keras.Sequential([
            tf.keras.Input(shape=(3)),
            tf.keras.layers.Dense(1)
        ])

        bayesian_model = svi.SVI(model,
                                 kl_scale=1.0,
                                 kl_fn=svi.mc_kl,
                                 prior_fn=svi.make_spike_and_slab_prior,
                                 posterior_fn=svi.make_mvn_posterior)
        bayesian_model.compile(loss=tfk.losses.MeanSquaredError(),
                               optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))
        bayesian_model.fit(_x, _y, epochs=2, batch_size=nn, verbose=1)


class Test64(TestSVI):
    def setUp(self) -> None:
        tf.keras.backend.set_floatx('float64')


if __name__ == '__main__':
    unittest.main()
