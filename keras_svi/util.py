import tensorflow as tf
from tensorflow.keras import backend as K

class PosteriorChecpoint(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super(PosteriorChecpoint, self).__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        ckpt = tf.train.Checkpoint(posterior_model=self.model.posterior.model)
        ckpt.write(self.path)


class PosteriorSummary(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super(PosteriorSummary, self).__init__()
        # self.path = path
        self.file_writer = tf.summary.create_file_writer(path)

    def on_epoch_end(self, epoch, logs=None):
        with self.file_writer.as_default():
            m_ = self.model.posterior.mean()
            s_ = self.model.posterior.stddev()

            for m, s, d in zip(m_, s_, self.model.posterior.model):
                names = d.name.split("_")
                tf.summary.histogram('/'.join(names + ['mean']), m, epoch)
                tf.summary.histogram('/'.join(names + ['stddev']), s, epoch)


class XTensorBoard(tf.keras.callbacks.TensorBoard):
    '''
    https://github.com/keras-team/keras/pull/9168#issuecomment-359901128
    '''

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super(XTensorBoard, self).on_epoch_end(epoch, logs)
