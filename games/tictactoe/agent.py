import numpy as np
import tensorflow as tf
from tensorflow import GradientTape, function
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam, SGD

from agents.q_network import QNetwork
from agents.policies import greedy, e_greedy
from agents.replay_buffer import ReplayBuffer
from agents.memory_buffer import MemoryBuffer

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 100

class DQL:

    def __init__(self, action_space, discount_factor=0.9, q_lr=0.001, target_lr=0.0001):

        self.q = QNetwork()
        self.q_target = QNetwork()
        self.discount_factor = DQL
        self.q_optimizer = Adam(q_lr)
        self.q_target_optimizer = SGD(target_lr)

        self.action_space = action_space

        self.loss = []

        self.train_policy = e_greedy
        self.play_policy = greedy

    def train(self, target):

        pass

    #@tf.function
    def predict(self, state, mask=None, explore=True):
        """

        :param state: Tensor None x state_size
        :param mask: Tensor None x action_size, bool
        :param explore:
        :return:
        """


        all_actions = tf.broadcast_to(tf.reshape(tf.linalg.set_diag(tf.zeros([self.action_space, self.action_space]),
                                         tf.fill([self.action_space], tf.constant(1, tf.float32))), (1, self.action_space, self.action_space)), [tf.shape(mask)[0], self.action_space, self.action_space])
        all_states = tf.broadcast_to(tf.reshape(state, (tf.shape(state)[0], 1, tf.shape(state)[1])), [tf.shape(mask)[0], self.action_space, tf.shape(state)[1]])

        if mask is not None:

            legal_actions = tf.ragged.boolean_mask(all_actions, mask)
            #legal_actions_idx =
            legal_states = tf.ragged.boolean_mask(all_states, mask)
        else:
            legal_actions = all_actions
            legal_states = all_states

        # get broadcast idx

        la_idx = legal_actions.nested_row_lengths()
        ls_idx = legal_states.nested_row_lengths()

        # transform to tensors again


        print(legal_actions)
        print(legal_states)

        legal_actions = legal_actions.merge_dims(inner_axis=1, outer_axis=0)
        legal_states = legal_states.merge_dims(inner_axis=1, outer_axis=0)

        print(legal_actions)
        print(legal_states)

        q_values = self.q_target.predict([legal_states, legal_actions])
        print(q_values)



        print(la_idx)
        # transform q-values to ragged tensor based on idx
        q_values = tf.RaggedTensor.from_nested_row_lengths(q_values, la_idx)
        legal_actions = tf.RaggedTensor.from_nested_row_lengths(legal_actions, ls_idx)

        print(q_values)

        def a_arg_max(inputs):

            q_values, actions = inputs
            idx = tf.argmax(q_values)
            opt_action = tf.reshape(tf.gather(actions, idx), (-1,))

            return opt_action

        print(legal_actions)
        optimal_idx = tf.map_fn(a_arg_max, elems=[q_values, legal_actions],
                                fn_output_signature=tf.TensorSpec(shape=[4,], dtype=tf.float32))

        # should replace this to policy part later
        print(optimal_idx)
        #

        #optimal_idx = self.train_policy(q_values) if explore else self.play_policy(q_values)

        print(optimal_idx)
        next_action = tf.reshape(optimal_idx, (-1,))
        # print(next_action)
        # optimal_action = tf.scatter_nd(legal_idx, next_action, tf.constant([9], dtype=tf.int64))
        # optimal_action = tf.reshape(optimal_action, (1, 9))

        # print(optimal_action)

        # self.previous_actions.append(optimal_actions)
        return next_action


beh = DQL(action_space=4)

state = tf.convert_to_tensor([[1, 0, 0, 0], [1, 0, 0, 1]], dtype=tf.float32)
mask = tf.convert_to_tensor([[True, False, False, True], [False, False, False, True]])


beh.predict(state=state, mask=mask, explore=True)

exit()


class AI:

    def __init__(self, agent_id, behavior, replay_buffer=True):

        self.id = agent_id
        self.behavior = behavior
        self.replay_buffer = ReplayBuffer() if replay_buffer else None
        self.memory_buffer = MemoryBuffer(['state', 'action', 'action_space'], 1)

        self.turn = 0
        self.episode = 0

    def play(self, state, mask, explore=True):
        """

        :param state:
        :param mask:
        :return:
        """

        self.turn += 1

        state = tf.convert_to_tensor(state)
        mask = tf.convert_to_tensor(mask)

        optimal_action = self.behavior.predict(state, mask, explore)
        self.memory_buffer.store(action=optimal_action)

        return optimal_action




    def observe(self, state, reward, action_space, done):

        if self.memory['state'] is not None:

            #print('------------------')
            #print(self.memory)
            #print(reward)
            #print(state)
            #print(action_space)
            #print(done)
            #print('----------------')
            #rom time import sleep
            #sleep(10)
            self.replay_buffer.add_sample(state=self.memory['state'],
                                          action=self.memory['action'],
                                          action_space=self.memory['action_space'],
                                          reward=reward,
                                          next_state=state,
                                          next_action_space=action_space,
                                          done=done)
            self._turn_rewards.append(reward)

        self.memory['state'] = state.copy()
        self.memory['action_space'] = action_space.copy()


        return

    def reset(self):

        if self.turn is not None:
            self.turn_hist.append(self.turn)
            self.reward_hist.append(sum(self._turn_rewards))
            self._turn_rewards = []
        self.turn = 0
        self.memory = {'state': None, 'action': None, 'action_space': None}

        return

    @tf.function
    def inner_update(self, states, actions, rewards, next_states, action_spaces, next_action_spaces, dones):

        # should find a way to make the update rule variable

        states = tf.cast(states, tf.float32)
        next_states = tf.cast(next_states, tf.float32)
        rewards = tf.cast(rewards, tf.float32)
        next_action_spaces = tf.cast(next_action_spaces, tf.float32)
        optimal_actions_list = []

        target_q_values = tf.zeros(shape=(0, 1), dtype=tf.float32)

        idx = tf.constant(0)
        def cond_loop(idx, tqv):
            return tf.less(idx, BATCH_SIZE)

        def while_body(idx, target_q_values):

            reward = tf.reshape(rewards[idx], (1, -1))
            done = dones[idx]
            if done:
                target_q_value = reward
            else:

                next_action_space = tf.reshape(next_action_spaces[idx], (-1,))

                legal_idx = tf.reshape(tf.where(tf.math.equal(next_action_space, tf.constant(0, dtype=tf.float32))), (-1, 1))
                next_state = tf.reshape(next_states[idx], (1, -1))

                next_actions_full = tf.zeros([9, 9])

                #print(next_action_space)
                next_actions_full = tf.linalg.set_diag(next_actions_full, tf.fill([9], tf.constant(1, tf.float32)))

                next_legal_actions = tf.reshape(tf.gather(next_actions_full, legal_idx), (-1, 9))

                optimal_states = tf.repeat(tf.reshape(next_state, (1, -1)), tf.shape(next_legal_actions)[0], axis=0)

                optimal_action = self.q_target([optimal_states, next_legal_actions])

                next_action = self.policy_greedy.sample(optimal_action)
                next_action = tf.reshape(next_action, (-1,))
                next_actions_full = tf.scatter_nd(legal_idx, next_action, tf.constant([9], dtype=tf.int64))
                next_actions_full = tf.reshape(next_actions_full, (1, 9))
                optimal_actions_list.append(next_actions_full)

                next_state = tf.reshape(next_state, [1, -1])

                next_actions_full = tf.cast(next_actions_full, tf.float32)
                target_q_value = reward + self.discount_factor * self.q([next_state, next_actions_full])


            target_q_values = tf.concat([target_q_values, target_q_value], axis=0)

            return (tf.add(idx, 1), target_q_values)

        target_q_values = tf.while_loop(cond_loop, while_body, [idx, target_q_values], shape_invariants=[idx.get_shape(), tf.TensorShape([None, 1])])[1]

        target_q_values = tf.reshape(target_q_values, (-1, 1))
        # compute td error

        actions = tf.squeeze(actions)
        with GradientTape() as tape:
            q_values = self.q([states, actions])

            q_loss = tf.reduce_mean(mean_squared_error(target_q_values, q_values))

        grads = tape.gradient(q_loss, self.q.trainable_variables)

        self.q_optimizer.apply_gradients(zip(grads, self.q.trainable_variables))

        # apply soft update
        grads = [(b - a) for a, b in zip(self.q.trainable_variables, self.q_target.trainable_variables)]
        self.q_target_optimizer.apply_gradients(zip(grads, self.q_target.trainable_variables))

        return tf.reduce_mean(q_loss)

    def update(self):

        if self.replay_buffer.current_size != 0:
            states, actions, rewards, next_states, action_spaces, next_action_spaces, dones = self.replay_buffer.sample(
                size=BATCH_SIZE)

            self.loss_hist.append(tf.reduce_mean(self.inner_update(states, actions, rewards, next_states, action_spaces, next_action_spaces, dones)))

        else:
            print('Didn\'t update because no samples available.')

        self.episodes += 1

        if self.episodes % 100 == 0:
            print(f'Average loss for last 100 updates: {sum(self.loss_hist[-100:])/len(self.loss_hist[-100:])}')

        return

    def save_model(self):

        self.q.save(f'./models/tictactoe/{self.id}/q')
        self.q_target.save(f'./models/tictactoe/{self.id}/q_target')

        return

    def load_model(self):

        self.q = tf.keras.models.load_model(f'./models/tictactoe/{self.id}/q_target')

        return


'''
beh = DQL()
ai = AI('1', beh)

state = np.array([[0, 0, 0, 1, 0]])
mask = np.array([0, 0, 1, 1])
print(ai.play(state, mask))

exit()
'''

