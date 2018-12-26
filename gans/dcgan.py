# -*- coding: utf-8 -*-
#
# gan.py
#

"""NOTES:
* Rescale and normalize input (3D) images.

"""

import tensorflow as tf


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
    def __init__(self):

        self.sess = None

        self._graph = None

    def build(self):

        samples = tf.placeholder(dtype=tf.float32, shape=[100], name='input')

        # self._graph ?

    def run(self, **kwargs):

        if self.sess is None:
            self.sess = tf.Session()

        return self.sess.run(
            graph=self._graph,
            feed_dict=kwargs,
            config=tf.ConfigProto(log_device_placement=True)
        )


class Discriminator:
    """
    Input: Tumor image of size (X x Y x Z).
    Output: Probability image is real/fake.
    """


# CHeckout: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
class DCGAN(Generator, Discriminator):
    """
    """


if __name__ == '__main__':
    # Defines initialization procedure.
    samples = tf.random_normal(shape=[100])
    # Create mutable object.
    x = tf.Variable(tf.zeros(dtype=tf.float32, shape=[100]), trainable=True)
    # Execute.
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(x))
