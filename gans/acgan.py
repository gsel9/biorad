# -*- coding: utf-8 -*-
#
# gan.py
#

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope


_logger = tf.logging._logger
_logger.setLevel(0)


def prelu(x, alpha=0.01):
    """Note that PReLU represents Leaky ReLU with alpha = 0.01.

    """

    return tf.where(tf.greater(x, 0), x, alpha * x)


def data_to_tensor(data_list, batch_size,  name=None):
    r"""Returns batch queues from the whole data.

    Args:
      data_list: A list of ndarrays. Every array must have the same size in the first dimension.
      batch_size: An integer.
      name: A name for the operations (optional).

    Returns:
      A list of tensors of `batch_size`.
    """
    # convert to constant tensor
    const_list = [tf.constant(data) for data in data_list]

    # create queue from constant tensor
    queue_list = tf.train.slice_input_producer(const_list, capacity=batch_size*128, name=name)

    # create batch queue
    return tf.train.shuffle_batch(queue_list, batch_size, capacity=batch_size*128,
        min_after_dequeue=batch_size*32, name=name)


class Discriminator:

    def build(self, tensor, num_category=10, batch_size=32, num_cont=2):

        reuse = len([t for t in tf.global_variables() if t.name.startswith('discriminator')]) > 0
        print(reuse)
        print(tensor.get_shape())
        with variable_scope.variable_scope('discriminator', reuse=reuse):
            tensor = slim.conv2d(tensor, num_outputs = 64, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)
            tensor = slim.conv2d(tensor, num_outputs=128, kernel_size=[4,4], stride=2, activation_fn=leaky_relu)
            tensor = slim.flatten(tensor)
            shared_tensor = slim.fully_connected(tensor, num_outputs=1024, activation_fn = leaky_relu)
            recog_shared = slim.fully_connected(shared_tensor, num_outputs=128, activation_fn = leaky_relu)
            disc = slim.fully_connected(shared_tensor, num_outputs=1, activation_fn=None)
            disc = tf.squeeze(disc, -1)
            recog_cat = slim.fully_connected(recog_shared, num_outputs=num_category, activation_fn=None)
            recog_cont = slim.fully_connected(recog_shared, num_outputs=num_cont, activation_fn=tf.nn.sigmoid)

        return disc, recog_cat, recog_cont


class Generator:

    def build(self, tensor):

        reuse = len([t for t in tf.global_variables() if t.name.startswith('generator')]) > 0
        print(tensor.get_shape())
        with variable_scope.variable_scope('generator', reuse = reuse):
            tensor = slim.fully_connected(tensor, 1024)
            print(tensor)
            tensor = slim.batch_norm(tensor, activation_fn=tf.nn.relu)
            tensor = slim.fully_connected(tensor, 7*7*128)
            tensor = slim.batch_norm(tensor, activation_fn=tf.nn.relu)
            tensor = tf.reshape(tensor, [-1, 7, 7, 128])
            # print '22',tensor.get_shape()
            tensor = slim.conv2d_transpose(tensor, 64, kernel_size=[4,4], stride=2, activation_fn = None)
            print('gen',tensor.get_shape())
            tensor = slim.batch_norm(tensor, activation_fn = tf.nn.relu)
            tensor = slim.conv2d_transpose(tensor, 1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.sigmoid)

        return tensor


class ACGAN:

    def train(self, num, x1, x2):

        target_num = tf.placeholder(dtype=tf.int32, shape=batch_size)
        target_cval_1 = tf.placeholder(dtype=tf.float32, shape=batch_size)
        target_cval_2 = tf.placeholder(dtype=tf.float32, shape=batch_size)

        z = tf.one_hot(tf.ones(batch_size, dtype=tf.int32) * target_num, depth=cat_dim)
        z = tf.concat(axis=z.get_shape().ndims-1, values=[z, tf.expand_dims(target_cval_1, -1), tf.expand_dims(target_cval_2, -1)])

        z = tf.concat(axis=z.get_shape().ndims-1, values=[z, tf.random_normal((batch_size, rand_dim))])

        gen = tf.squeeze(generator(z), -1)

        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                          tf.local_variables_initializer()))
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint('checkpoint_dir'))
            imgs = sess.run(gen, {target_num: num, target_cval_1: x1, target_cval_2:x2})

            _, ax = plt.subplots(10,10, sharex=True, sharey=True)
            for i in range(10):
                for j in range(10):
                    ax[i][j].imshow(imgs[i*10+j], 'gray')
                    ax[i][j].set_axis_off()

        a = np.random.randint(0, cat_dim, batch_size)
        run_generator(a,
                      np.random.uniform(0, 1, batch_size), np.random.uniform(0, 1, batch_size),
        fig_name='fake.png')

        # classified image

train(
    np.arange(10).repeat(10), np.linspace(0, 1, 10).repeat(10),
    np.expand_dims(np.linspace(0, 1, 10), axis=1).repeat(10, axis=1).T.flatten(),
    )
