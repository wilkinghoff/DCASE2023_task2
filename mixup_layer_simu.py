from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_probability as tfp

class MixupLayer(layers.Layer):
    def __init__(self, prob, alpha=1, **kwargs):
        super(MixupLayer, self).__init__(**kwargs)
        self.prob = prob
        self.alpha = alpha

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, training=None):
        # get mixup weights
        if self.alpha == 1:
            l = tf.random.uniform(shape=[tf.shape(inputs[0])[0]])*0.5
        X_l = tf.reshape(l, [-1]+[1]*(len(inputs[0].shape)-1))
        y_l = tf.reshape(l, [-1]+[1]*(len(inputs[1].shape)-1))

        # mixup data
        X1 = inputs[0]
        X2 = tf.reverse(inputs[0], axis=[0])
        X = X1 * X_l + X2 * (1 - X_l)

        # mixup labels
        y1 = inputs[1]
        y2 = tf.reverse(inputs[1], axis=[0])
        y = y1 * y_l + y2 * (1 - y_l)
        y_new = tf.concat([tf.zeros_like(y), y], axis=-1)
        p = tf.random.uniform(shape=[tf.shape(inputs[0])[0]]) * 0.5
        y_p = tf.reshape(p, [-1]+[1]*(len(inputs[1].shape)-1))
        N = inputs[1].shape[1]
        y1_new = tf.concat([y1, tf.zeros_like(y)], axis=-1)*(1-y_p-y_p/(N-1))+y_p/(N-1)

        # apply mixup or not
        dec = tf.dtypes.cast(tf.random.uniform(shape=[tf.shape(inputs[0])[0]]) < self.prob, tf.dtypes.float32)
        dec1 = tf.reshape(dec, [-1] + [1] * (len(inputs[0].shape) - 1))
        out1 = dec1 * X + (1 - dec1) * X1
        dec2 = tf.reshape(dec, [-1] + [1] * (len(y_new.shape) - 1))
        out2 = dec2 * y_new + (1 - dec2) * y1_new
        outputs = [out1, out2]

        # pick output corresponding to training phase
        return K.in_train_phase(outputs, [inputs[0], tf.concat([inputs[1], tf.zeros_like(inputs[1])], axis=-1)], training=training)

    def get_config(self):
        config = {
            'prob': self.prob,
            'alpha': self.alpha
        }
        base_config = super(MixupLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

