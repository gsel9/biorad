# -*- coding: utf-8 -*-
#
# gan.py
#

"""NOTES:
* Rescale and normalize input (3D) images.

"""

import tensorflow as tf


# Checkout: https://towardsdatascience.com/implementing-a-generative-adversarial-network-gan-dcgan-to-draw-human-faces-8291616904a
class Generator:
    """
    Input: Tensor of 100 random numbers.
    Output: Tumor image of size (X x Y x Z).

    Note:
    - Normalizing responses to have zero mean and unit variance over the entire
      mini-batch  stabilizes  the  GAN  learning  process  and  prevents the
      generator from collapsing all samples to a single point (see: S. Ioffe
      and C. Szegedy, “Batch normalization: Accelerating deep network training
      by reducing internal covariate shift,” in International Conference on
      Machine Learning, 2015, pp. 448–456).
    -

    """

    @staticmethod
    def prelu(x, param=0.01):
        """Parameter Rectified Linear Unit."""

        # With param=0.01: PReLU = LeakyReLU.
        return tf.maximum(x, tf.multiply(x, param))

    def build(z, momentum=0.99, keep_prob=keep_prob, is_training=is_training):

        with tf.variable_scope("generator", reuse=None):
            x = z
            d1 = 4
            d2 = 1
            x = tf.layers.dense(x, units=d1 * d1 * d2, activation=activation)
            x = tf.layers.dropout(x, keep_prob)
            x = tf.contrib.layers.batch_norm(
                x, is_training=is_training, decay=momentum
            )
            x = tf.reshape(x, shape=[-1, d1, d1, d2])
            x = tf.image.resize_images(x, size=[7, 7])
            x = tf.layers.conv2d_transpose(
                x,
                kernel_size=5, filters=64, strides=2, padding='same',
                activation=activation
            )
            x = tf.layers.dropout(x, keep_prob)
            x = tf.contrib.layers.batch_norm(
                x, is_training=is_training, decay=momentum
            )
            x = tf.layers.conv2d_transpose(
                x,
                kernel_size=5, filters=64, strides=2, padding='same',
                activation=activation
            )
            x = tf.layers.dropout(x, keep_prob)
            x = tf.contrib.layers.batch_norm(
                x, is_training=is_training, decay=momentum
            )
            x = tf.layers.conv2d_transpose(
                x,
                kernel_size=5, filters=64, strides=1, padding='same',
                activation=activation
            )
            x = tf.layers.dropout(x, keep_prob)
            x = tf.contrib.layers.batch_norm(
                x, is_training=is_training, decay=momentum
            )
            x = tf.layers.conv2d_transpose(
                x,
                kernel_size=5, filters=1, strides=1, padding='same',
                activation=tf.nn.sigmoid
            )
        return x


# Checkout: https://towardsdatascience.com/implementing-a-generative-adversarial-network-gan-dcgan-to-draw-human-faces-8291616904a
class Discriminator:
    """
    Input: Tumor image of size (X x Y x Z).
    Output: Probability image is real/fake.
    """

    @staticmethod
    def prelu(x, param=0.01):
        """Parameter Rectified Linear Unit."""

        # With param=0.01: PReLU = LeakyReLU.
        return tf.maximum(x, tf.multiply(x, param))

    def build(self, img_in, reuse=None, keep_prob=keep_prob):

        with tf.variable_scope("discriminator", reuse=reuse):
            x = tf.reshape(img_in, shape=[-1, 28, 28, 1])
            x = tf.layers.conv2d(
                x,
                kernel_size=5, filters=64, strides=2, padding='same',
                activation=self.prelu
            )
            x = tf.layers.dropout(x, keep_prob)
            x = tf.layers.conv2d(
                x,
                kernel_size=5, filters=64, strides=1, padding='same',
                activation=self.prelu
            )
            x = tf.layers.dropout(x, keep_prob)
            x = tf.layers.conv2d(
                x,
                kernel_size=5, filters=64, strides=1, padding='same',
                activation=self.prelu
            )
            x = tf.layers.dropout(x, keep_prob)
            x = tf.contrib.layers.flatten(x)
            x = tf.layers.dense(x, units=128, activation=self.prelu)
            x = tf.layers.dense(x, units=1, activation=tf.nn.sigmoid)
        return x




# CHeckout: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
class DCGAN(Generator, Discriminator):
    """
    """

    @staticmethod
    def binary_cross_entropy(x, z, eps=1e-12):
        """"""

        return (-1.0 * (x * tf.log(z + eps) + (1. - x) * tf.log(1. - z + eps)))

    def build(X_in, noise, keep_prob, is_training, learning_rate=0.00015):

        g = generator(noise, keep_prob, is_training)
        d_real = discriminator(X_in)
        d_fake = discriminator(g, reuse=True)

        vars_g = [
            var for var in tf.trainable_variables()
            if var.name.startswith('generator')
        ]
        vars_d = [
            var for var in tf.trainable_variables()
            if var.name.startswith('discriminator')
        ]

        d_reg = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(1e-6), vars_d
        )
        g_reg = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(1e-6), vars_g
        )
        loss_d_real = binary_cross_entropy(tf.ones_like(d_real), d_real)
        loss_d_fake = binary_cross_entropy(tf.zeros_like(d_fake), d_fake)
        loss_g = tf.reduce_mean(
            self.binary_cross_entropy(tf.ones_like(d_fake), d_fake)
        )
        loss_d = tf.reduce_mean(0.5 * (loss_d_real + loss_d_fake))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer_d = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate
            ).minimize(loss_d + d_reg, var_list=vars_d)
            optimizer_g = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate
            ).minimize(loss_g + g_reg, var_list=vars_g)

    def noise_generator(self, batch_size, n_noise):

        n = np.random.uniform(0.0, 1.0, [batch_size, n_noise], dtype=np.float32)
        batch = [
            np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]
        ]
        return batch

    def train(self, num_epochs=60000):

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())


        for i in range(num_epochs):
            train_d = True
            train_g = True
            keep_prob_train = 0.6 # 0.5

            noise = self.noise_generator(batch_size, n_noise)

            d_real_ls, d_fake_ls, g_ls, d_ls = sess.run(
                [loss_d_real, loss_d_fake, loss_g, loss_d],
                feed_dict={
                    X_in: batch,
                    noise: n,
                    keep_prob: keep_prob_train,
                    is_training:True
                }
            )
            d_real_ls = np.mean(d_real_ls)
            d_fake_ls = np.mean(d_fake_ls)
            g_ls = g_ls
            d_ls = d_ls

            if g_ls * 1.5 < d_ls:
                train_g = False
                pass
            if d_ls * 2 < g_ls:
                train_d = False
                pass

            if train_d:
                sess.run(
                    optimizer_d,
                    feed_dict={
                        noise: n,
                        X_in: batch,
                        keep_prob: keep_prob_train,
                        is_training:True
                    }
                )
            if train_g:
                sess.run(
                    optimizer_g,
                    feed_dict={
                        noise: n,
                        keep_prob: keep_prob_train,
                        is_training:True
                    }
                )


if __name__ == '__main__':

    tf.reset_default_graph()

    X_in = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28], name='X')
    noise = tf.placeholder(dtype=tf.float32, shape=[None, n_noise])

    keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
    is_training = tf.placeholder(dtype=tf.bool, name='is_training')

    # For tensorflow to apply batch normalization, we need to let it know whether we are in training mode.
    # The keep_prob variable will be used by our dropout layers, which we introduce for more stable learning outcomes.
