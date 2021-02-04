import numpy as np
import tensorflow as tf
from tensorflow import GradientTape, function
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.optimizers import Adam, SGD

from agents.q_network import QNetwork
from agents.policies import EGreedy, Greedy, Greedy2
from agents.replay_buffer import ReplayBuffer

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

BATCH_SIZE = 100


class AI:

    def __init__(self, agent_id):

        self.id = agent_id

        # model params - this part needs to be rewritten, so we can dynamically choose the config params
        self.q = QNetwork()
        self.q_target = QNetwork()
        self.discount_factor = 0.9
        self.policy = EGreedy(epsilon=0.1)
        self.policy_greedy = Greedy()
        self.policy_greedy2 = Greedy2()
        self.q_optimizer = Adam(0.001)
        self.q_target_optimizer = SGD(0.001)

        # not sure if this should be list or numpy/tensor
        # this is what we need to keep track of the environment to let our agent reflect and learn from its interactions

        # replay buffer still needs to be implemented and is currently not in use
        self.replay_buffer = ReplayBuffer()
        self.use_rb = False

        self.turn = None

        # metrics
        self.loss_hist = []
        self.reward_hist = []
        self.turn_hist = []

        self.episodes = 0

        self._turn_rewards = []

        self.memory = {'state': None, 'action': None, 'action_space': None}

    def play(self, state, legal_actions, greedy=False, target=False):

        self.turn += 1

        greedy = tf.convert_to_tensor(greedy)
        target = tf.convert_to_tensor(target)

        state = tf.convert_to_tensor(state)
        legal_actions = tf.convert_to_tensor(legal_actions)

        optimal_action = self.inner_play(state, legal_actions, greedy, target)
        self.memory['action'] = optimal_action
        optimal_action = optimal_action.numpy().reshape([9])
        return optimal_action

    @tf.function
    def inner_play(self, state, legal_actions, greedy, target):

        #legal_idx = tf.reshape(tf.where(tf.math.equal(legal_actions, tf.constant(0, dtype=tf.float64))), (-1, 1))

        all_actions = tf.linalg.set_diag(tf.zeros([9, 9]), tf.fill([9], tf.constant(1, tf.float32)))

        #legal_action = tf.reshape(tf.gather(next_actions_full, legal_idx), (-1, 9))

        state = tf.repeat(tf.reshape(state, (1, -1)), 9, axis=0)

        def true_return():
            return self.q_target([state, all_actions])

        def false_return():
            return self.q([state, all_actions])

        optimal_actions = tf.cond(target, true_return, false_return)

        print(optimal_actions)
        print(tf.reshape(legal_actions, (1, 9)))


        # loop over legal actions and select action with highest q-value\
        #if target:
        #   optimal_actions = self.q_target([state, legal_actions])
        #else:
        #   optimal_actions = self.q([state, legal_actions])

        # print(optimal_actions)

        def true_return():
            return self.policy_greedy2.sample(optimal_actions, legal_actions)

        def false_return():
            return self.policy.sample(optimal_actions, legal_actions)

        optimal_idx = tf.cond(greedy, true_return, false_return)

        #if greedy:
        #   optimal_idx = self.policy_greedy.sample(optimal_actions)
        #else:
        #   optimal_idx = self.policy.sample(optimal_actions)

        # print(optimal_idx)

        print(optimal_idx)
        next_action = tf.reshape(optimal_idx, (-1,))
        # print(next_action)
        #optimal_action = tf.scatter_nd(legal_idx, next_action, tf.constant([9], dtype=tf.int64))
        #optimal_action = tf.reshape(optimal_action, (1, 9))


        #print(optimal_action)

        # self.previous_actions.append(optimal_actions)
        return next_action


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




