r"""
Test nn_utils.py
"""


import unittest

import numpy as np
import tensorflow as tf

import nn_utils


class TestBinarize(unittest.TestCase):
  """ Test the binarize function """

  def test_forward(self):
    """ Test the implementation of the forward function.

    Input tensor has all its values within range [-128, 128].
    """

    with tf.Graph().as_default():
      x = tf.placeholder(tf.float32, shape=[32])
      y = nn_utils.binarize(x)

      with tf.Session() as sess:
        x_val = np.random.random(32) * 256 - 128
        y_val = sess.run(y, feed_dict={x: x_val})

      for i in range(len(x_val)):
        if x_val[i] < 0:
          self.assertEqual(y_val[i], - 1.0)
        elif x_val[i] > 0:
          self.assertEqual(y_val[i], 1.0)
        else:
          self.assertEqual(y_val[i], 0.0)

  def test_backward(self):
    """ Test the gradient of the binarize function.

    For a clip function, if x is within the clipping range,
    its gradient should be 1.0.
    Or the gradient is 0.0 (think about the gradient of a constant).

    We test the property above by giving a random array in
    the value range [-2.0, 2.0].
    """

    with tf.Graph().as_default():
      x = tf.placeholder(tf.float32, shape=[32])
      y = nn_utils.binarize(x)

      # Get the gradient of y w.r.t x.
      grad = tf.gradients(y, [x])[0]

      with tf.Session() as sess:
        x_val = np.random.random(32) * 4 - 2.0
        _, grad_val = sess.run([y, grad], feed_dict={x: x_val})

      for i in range(len(x_val)):
        if x_val[i] <= 1.0 and x_val[i] >= -1.0:
          self.assertEqual(grad_val[i], 1.0)
        else:
          self.assertEqual(grad_val[i], 0.0,
                           'The gradient of %f should be 0.0, not %f'
                           % (x_val[i], grad_val[i]))


class TestStochasticBinarize(unittest.TestCase):
  """ Test the stochastic_binarize function """

  def test_forward(self):
    """ Test the implementation of the forward function.
    """

    with tf.Graph().as_default():
      x = tf.placeholder(tf.float32, shape=[32])
      y, probs = nn_utils.stochastic_binarize(x)

      with tf.Session() as sess:
        x_val = np.random.random(32) * 4 - 2
        y_val, probs_val = sess.run([y, probs], feed_dict={x: x_val})

      for i in range(len(x_val)):
        if x_val[i] >= -1 and x_val[i] <= 1:
          self.assertAlmostEqual(probs_val[i], (x_val[i] + 1) / 2)
        elif x_val[i] >= 1:
          self.assertEqual(probs_val[i], 1.0)
        else:
          self.assertEqual(probs_val[i], 0.0)

  def test_backward(self):
    """ Evaluate the gradient of stochastic binarization. """

    with tf.Graph().as_default():
      x = tf.placeholder(tf.float32, shape=[32])
      y, _ = nn_utils.stochastic_binarize(x)

      # Get the gradient of y w.r.t x.
      # and this gradient should be None since there is no direct
      # relationship between y and x.
      grad = tf.gradients(y, [x])[0]

      self.assertIsNone(grad)


if __name__ == '__main__':
  unittest.main()
