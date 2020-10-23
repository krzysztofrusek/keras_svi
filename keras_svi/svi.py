import tensorflow as tf
import tensorflow_probability as tfp
tfk = tf.keras
tfd = tfp.distributions
tfb = tfp.bijectors


def _make_posterior(v):
    n = len(v.shape)
    return tfd.Independent(tfd.Normal(loc=tf.Variable(tf.convert_to_tensor(v)),
                                      scale=tfp.util.TransformedVariable(0.1 + tf.zeros_like(v),
                                                                         tfb.Softplus()(tfb.Scale(0.2)))),
                           reinterpreted_batch_ndims=n)


def _make_prior(posterior):
    n = len(posterior.event_shape)
    return tfd.Independent(tfd.Normal(tf.zeros(posterior.event_shape), 2.),
                           reinterpreted_batch_ndims=n)

#TODO Handle Fused RNN
class SVI(tfk.Model):
    def __init__(self, model, kl_scale=1.0,
                 posterior_fn=_make_posterior,
                 prior_fn = _make_prior):
        super(SVI, self).__init__()

        self.model = model
        self.kl_scale=kl_scale
        self.posterior_fn = posterior_fn
        self.prior_fn = prior_fn

    def build(self, input_shape):
        self.model.build(input_shape)
        tf.print('build')

        vars = self.model.trainable_variables
        self.posterior = tfp.distributions.JointDistributionSequential([self.posterior_fn(v) for v in vars])
        self.prior = tfp.distributions.JointDistributionSequential([
            self.prior_fn(m) for m in self.posterior.model
        ])
        self.vars = vars
        super(SVI, self).build(input_shape)

    def call(self,inputs):
        theta = self.posterior.sample()
        for v, s in zip(self.vars, theta):
            v.assign(s)
        return self.model(inputs)


    def train_step(self,data):
        x,y = data

        #trigger creation of posterior
        if not self.built:
            self(x)

        with tf.GradientTape() as tape:

            with tape.stop_recording():
                with tf.GradientTape() as sample_tape:
                    theta = self.posterior.sample()
                sample_grad = sample_tape.gradient(theta, self.posterior.variables)

            for v, s in zip(self.vars, theta):
                v.assign(s)
            yhat = self.model(x)

            with tape.stop_recording():
                with tf.GradientTape() as kl_tape:
                    kl = self.posterior.kl_divergence(self.prior)
                kl_grad = kl_tape.gradient(kl, self.posterior.variables)

            loss = self.compiled_loss(y, yhat)

        grad = tape.gradient(loss, self.vars)

        grad_pairs =[]

        for g in grad:
            grad_pairs.append(g)
            # Here I assume that all posterios surageate are from the same familly
            if not isinstance(self.posterior.model[0].distribution, tfd.Deterministic):
                grad_pairs.append(g)

        elbo = loss + self.kl_scale * kl
        elbo_grad = [g1 * g2 + self.kl_scale*g3 for g1, g2, g3 in zip(grad_pairs, sample_grad, kl_grad)]

        self.optimizer.apply_gradients(zip(elbo_grad, self.posterior.variables))

        self.compiled_metrics.update_state(y, yhat)
        # Return a dict mapping metric names to current value
        ret = {m.name: m.result() for m in self.metrics}
        ret.update(dict(loss=loss, kl=kl, elbo=elbo))
        return ret

    def predict_step(self, data):
        theta = self.posterior.sample()

        for v, s in zip(self.vars, theta):
            v.assign(s)
        yhat = self.model(data)
        return yhat




