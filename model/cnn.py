import numpy as np

import tensorflow as tf

class Simple_CNN(object):

    def __init__(self, X, num_classes):

        self.keep_prob = tf.placeholder(tf.float32, name="drop_out_keep_prob")

        # (None, 224, 224, 3) -> (32, 224, 224, 3)
        W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
        L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
        L1 = tf.nn.relu(L1)
        # (32, 224, 224, 3) --> (32, 224, 224, 3)
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 32], stddev=0.01))
        L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
        L2 = tf.nn.relu(L2)
        # (32, 224, 224, 3) --> (64, 112, 112, 3)
        L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L2 = tf.nn.dropout(L2, self.keep_prob)

        # (64, 112, 112, 3) --> (64, 112, 112, 3)
        W3 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
        L3 = tf.nn.relu(L3)
        # (64, 112, 112, 3) --> (64, 112, 112, 3)
        W4 = tf.Variable(tf.random_normal([3, 3, 64, 64], ))
        L4 = tf.nn.conv2d(L3, W4, strides=[1, 1, 1, 1], padding='SAME')
        L4 = tf.nn.relu(L4)
        # (64, 112, 112, 3) --> (64, 56, 56, 3)
        L4 = tf.nn.max_pool(L4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        L4 = tf.nn.dropout(L4, self.keep_prob)

        W5 = tf.Variable(tf.random_normal([56 * 56 * 64, 512], stddev=0.01))
        L5 = tf.reshape(L4, [-1, 56 * 56 * 64])
        L5 = tf.matmul(L5, W5)
        L5 = tf.nn.relu(L5)
        L5 = tf.nn.dropout(L5, self.keep_prob)

        W6 = tf.Variable(tf.random_normal([512, 100], stddev=0.01), name="spatial_squeeze")

        self.logits = tf.matmul(L5, W6)


