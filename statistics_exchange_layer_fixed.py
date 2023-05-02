from tensorflow.keras import backend as K
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow_probability as tfp

class XchangeLayer(layers.Layer):
    def __init__(self, prob, num_classes, alpha=1, **kwargs):
        super(XchangeLayer, self).__init__(**kwargs)
        self.prob = prob
        self.alpha = alpha
        self.num_classes = 1#num_classes

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, training=None):
        # get mixup weights
        if self.alpha == 1:
            #dist = tfp.distributions.Beta(0.5, 0.5)
            #l = dist.sample([tf.shape(inputs[0])[0]])
            l = tf.random.uniform(shape=[tf.shape(inputs[0])[0]])
        X_l = tf.reshape(l, [-1]+[1]*(len(inputs[0].shape)-1))
        XX_l = tf.reshape(l, [-1]+[1]*(len(inputs[1].shape)-1))
        y_l = tf.reshape(l, [-1]+[1]*(len(inputs[2].shape)-1))

        # mixup data
        X1 = inputs[0]
        X2 = tf.reverse(inputs[0], axis=[0])
        X = X1 * X_l + X2 * (1 - X_l)
        XX1 = inputs[1]
        XX2 = tf.reverse(inputs[1], axis=[0])
        XX = XX1 * XX_l + XX2 * (1 - XX_l)

        # mixup labels
        y1 = inputs[2]
        y2 = tf.reverse(inputs[2], axis=[0])
        y = y1 * y_l + y2 * (1 - y_l)
        #y = tf.math.maximum(y1 * y_l, y2 * (1 - y_l))
        y_new = tf.concat([y1, tf.repeat(tf.zeros_like(y1), self.num_classes, axis=-1)], axis=-1)

        # statistics exchange data
        X_tex = (X1-tf.math.reduce_mean(X1, axis=2, keepdims=True))/(tf.math.reduce_std(X1, axis=2, keepdims=True)+1e-16)*tf.math.reduce_std(X2, axis=2, keepdims=True)+tf.math.reduce_mean(X2, axis=2, keepdims=True)
        y_tex = tf.concat([tf.zeros_like(y1), y1#tf.repeat(y1, self.num_classes, axis=-1)*tf.tile(y1, [1, self.num_classes])#, tf.repeat(tf.zeros_like(y1), self.num_classes, axis=-1)
                           ], axis=-1)

        X_fex = (X1-tf.math.reduce_mean(X1, axis=1, keepdims=True))/(tf.math.reduce_std(X1, axis=1, keepdims=True)+1e-16)*tf.math.reduce_std(X2, axis=1, keepdims=True)+tf.math.reduce_mean(X2, axis=1, keepdims=True)
        y_fex = tf.concat([tf.zeros_like(y1),# tf.repeat(tf.zeros_like(y1), self.num_classes, axis=-1),
                           y1#tf.repeat(y1, self.num_classes, axis=-1)*tf.tile(y1, [1, self.num_classes])
                           ], axis=-1)

        # randomly decide on which statistics exchange axis to use
        dec = tf.dtypes.cast(tf.random.uniform(shape=[tf.shape(inputs[0])[0]]) < 0.5, tf.dtypes.float32)
        dec1 = tf.reshape(dec, [-1] + [1] * (len(inputs[0].shape) - 1))
        X_ex = dec1 * X_fex + (1 - dec1) * X_tex
        dec2 = tf.reshape(dec, [-1] + [1] * (len(y_new.shape) - 1))
        y_ex = dec2 * y_fex + (1 - dec2) * y_tex

        # apply mixup or not
        dec = tf.dtypes.cast(tf.random.uniform(shape=[tf.shape(inputs[0])[0]]) < self.prob, tf.dtypes.float32)
        dec1 = tf.reshape(dec, [-1] + [1] * (len(inputs[0].shape) - 1))
        out1 = dec1 * X + (1 - dec1) * X_ex
        #dec2 = tf.reshape(dec, [-1] + [1] * (len(inputs[1].shape) - 1))
        out2 = XX#dec2 * XX + (1 - dec2) * XX1
        dec3 = tf.reshape(dec, [-1] + [1] * (len(y_new.shape) - 1))
        out3 = dec3 * y_new + (1 - dec3) * y_ex
        outputs = [out1, out2, out3]

        # pick output corresponding to training phase
        return K.in_train_phase(outputs, [inputs[0], inputs[1], tf.concat([inputs[2], tf.repeat(tf.zeros_like(inputs[2]), self.num_classes, axis=-1)], axis=-1)], training=training)

    def get_config(self):
        config = {
            'prob': self.prob,
            'num_classes': self.num_classes,
            'alpha': self.alpha
        }
        base_config = super(XchangeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

