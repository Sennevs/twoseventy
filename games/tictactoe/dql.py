import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf

from agents.q_network import QNetwork
from agents.policies import egreedy_ragged, greedy_ragged


class DQL:

    def __init__(self, action_space, discount_factor=0.9, q_lr=0.001, target_lr=0.0001):

        self.q = QNetwork()
        self.q_target = QNetwork()
        self.discount_factor = discount_factor
        self.q_optimizer = tf.optimizers.Adam(q_lr)
        self.q_target_optimizer = tf.optimizers.SGD(target_lr)

        self.action_space = action_space

        self.loss = []

        self.train_policy = egreedy_ragged
        self.play_policy = greedy_ragged

    def train(self, target):

        pass

    #@tf.function
    def predict(self, state, action_mask=None, explore=True):

        """

        :param state: Tensor None x state_size
        :param action_mask: Tensor None x action_size, bool
        :param explore:
        :return:
        """

        print('hello')
        print(state)
        print(action_mask)

        all_actions = tf.broadcast_to(tf.reshape(tf.linalg.set_diag(tf.zeros([self.action_space, self.action_space]),
                                         tf.fill([self.action_space], tf.constant(1, tf.float32))), (1, self.action_space, self.action_space)), [tf.shape(action_mask)[0], self.action_space, self.action_space])
        all_states = tf.broadcast_to(tf.reshape(state, (tf.shape(state)[0], 1, tf.shape(state)[1])), [tf.shape(action_mask)[0], self.action_space, tf.shape(state)[1]])


        legal_actions = all_actions if action_mask is None else tf.ragged.boolean_mask(all_actions, action_mask)
        legal_states = all_actions if action_mask is None else tf.boolean_mask(all_states, action_mask)

        # transform to tensors again
        legal_actions_tensor = legal_actions.merge_dims(inner_axis=1, outer_axis=0)

        legal_states = tf.reshape(legal_states, (-1, tf.shape(state)[1]))
        legal_actions_tensor = tf.reshape(legal_actions_tensor, (-1, self.action_space))

        print(legal_states)
        print(legal_actions_tensor)
        q_values = self.q_target([legal_states, legal_actions_tensor])

        # transform q-values to ragged tensor based on idx
        q_values = tf.RaggedTensor.from_nested_row_lengths(q_values, legal_actions.nested_row_lengths())

        if explore:
            optimal_action = tf.map_fn(egreedy_ragged, elems=[q_values, legal_actions],
                                       fn_output_signature=tf.TensorSpec(shape=[self.action_space, ], dtype=tf.float32))
        else:
            optimal_action = tf.map_fn(greedy_ragged, elems=[q_values, legal_actions],
                                       fn_output_signature=tf.TensorSpec(shape=[self.action_space, ], dtype=tf.float32))

        return optimal_action

    #@tf.function
    def train(self, states, actions, rewards, next_states, action_masks, next_action_masks, dones):

        rewards = tf.cast(rewards, tf.float32)

        dones_mask = tf.reshape(dones, (-1,))
        not_dones_mask = tf.logical_not(dones_mask)
        rewards_d = tf.boolean_mask(rewards, dones_mask, axis=0)
        rewards_nd = tf.boolean_mask(rewards, not_dones_mask, axis=0)
        next_states_nd = tf.boolean_mask(next_states, not_dones_mask, axis=0)
        next_action_masks_nd = tf.boolean_mask(next_action_masks, not_dones_mask, axis=0)


        target_q_value_d = rewards_d

        print(target_q_value_d)


        next_actions_nd = self.predict(next_states_nd, next_action_masks_nd)

        print(['a:'])
        print(next_actions_nd)

        rewards_nd = tf.reshape(tf.cast(rewards_nd, tf.float32), (-1, 1))
        print(next_states_nd)
        target_q_value_nd = rewards_nd + self.discount_factor * self.q([next_states_nd, next_actions_nd])

        print(rewards_nd)
        print(self.discount_factor)
        print(next_actions_nd)

        if tf.shape(target_q_value_d)[0] == 0:
            print('a')
            target_q_values = target_q_value_nd
        elif tf.shape(target_q_value_nd)[0] == 0:
            print('b')

            target_q_values = target_q_value_d
        else:
            print('c')

            print(target_q_value_nd)
            # combine them again
            done_idx = tf.cast(tf.where(dones_mask), tf.int32)
            not_done_idx = tf.cast(tf.where(not_dones_mask), tf.int32)
            t_foo = tf.scatter_nd(done_idx, tf.reshape(target_q_value_d, (-1, 1)), shape=(tf.shape(actions)[0], 1))
            f_foo = tf.scatter_nd(not_done_idx, tf.reshape(target_q_value_nd, (-1, 1)), shape=(tf.shape(actions)[0], 1))
            target_q_values = t_foo + f_foo

        print(states)
        print(actions)
        # compute td error
        with tf.GradientTape() as tape:
            q_values = self.q([states, actions])
            q_loss = tf.reduce_mean(tf.keras.losses.mean_squared_error(target_q_values, q_values))

        grads = tape.gradient(q_loss, self.q.trainable_variables)

        self.q_optimizer.apply_gradients(zip(grads, self.q.trainable_variables))

        # apply soft update
        grads = [(b - a) for a, b in zip(self.q.trainable_variables, self.q_target.trainable_variables)]
        self.q_target_optimizer.apply_gradients(zip(grads, self.q_target.trainable_variables))

        return tf.reduce_mean(q_loss)

    def save_model(self, agent_id):

        self.q.save(f'./models/tictactoe/{agent_id}/q')
        self.q_target.save(f'./models/tictactoe/{agent_id}/q_target')

        return

    def load_model(self, agent_id):

        self.q = tf.keras.models.load_model(f'./models/tictactoe/{agent_id}/q_target')

        return



'''
beh = DQL()
ai = AI('1', beh)

state = np.array([[0, 0, 0, 1, 0]])
mask = np.array([0, 0, 1, 1])
print(ai.play(state, mask))

exit()
'''


'''

beh = DQL(action_space=4)

states = tf.convert_to_tensor([[1, 0, 0, 0], [1, 0, 0, 1]], dtype=tf.float32)
actions = tf.convert_to_tensor([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=tf.float32)
rewards = tf.convert_to_tensor([[1], [0]], dtype=tf.float32)
next_states = tf.convert_to_tensor([[1, 0, 0, 1], [1, 1, 0, 1]], dtype=tf.float32)
action_masks = tf.convert_to_tensor([[True, False, False, True], [False, False, False, True]])
next_action_masks = tf.convert_to_tensor([[True, False, False, True], [False, False, False, True]])
dones = tf.convert_to_tensor([[True], [False]])


print(beh.train(states, actions, rewards, next_states, action_masks, next_action_masks, dones))

#print(beh.predict(state=state, mask=mask, explore=True))

exit()'''
