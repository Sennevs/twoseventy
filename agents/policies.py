import random
import numpy as np
import tensorflow as tf

import tensorflow_probability as tfp

def uniform(actions, legal_actions=None):
    """

    :param actions:
    :param legal_actions:
    :return:
    """

    #legal_values = tf.gather(values, action_filter) if action_filter is None else values

    legal_actions = legal_actions or actions
    opt_idx = tf.random.uniform(shape=(1,), minval=0, maxval=tf.shape(legal_actions)) if legal_actions is None else 0
    opt_action = tf.gather(legal_actions, opt_idx)



    return opt_action

#actions = tf.Tensor([1, 1, 1])
#legal_actions = tf.Tensor()




def e_greedy(epsilon, actions, filter=None):
    """

    :param q_function:
    :param state:
    :param action_space:
    :return:
    """

    greedy = random.random() > epsilon

    if greedy:
        idx = tf.argmax(optimal_probs, axis=0)
    else:
        print('Just did a rando')

        s = tfp.distributions.Normal(1, 1)
        samples = tf.abs(s.sample((9,1)))

        optimal_probs *= samples

        idx = tf.argmax(optimal_probs, axis=0)

    action = tf.one_hot(idx, tf.shape(actions)[0])

    return action


def greedy(filter=None):
        """

        :param q_function:
        :param state:
        :param action_space:
        :return:
        """

        optimal_probs = (tf.exp(actions) / tf.reduce_sum(tf.exp(actions))) * tf.cast(
            tf.reshape((legal_actions - 1) * -1, (9, 1)), dtype=tf.float32)

        action = tf.one_hot(tf.argmax(optimal_probs, axis=0), 9)

        return action



