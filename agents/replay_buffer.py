from collections import deque
import random


class ReplayBuffer:

    def __init__(self, max_size=100):

        self.states = deque()
        self.actions = deque()
        self.rewards = deque()
        self.next_states = deque()

        self.action_spaces = deque()
        self.next_action_spaces = deque()

        self.max_size = max_size

        self.current_size = 0

        return

    def sample(self, size):

        samples = [random.randrange(self.current_size) for _ in range(size)]

        return_states = [self.states[i] for i in samples]
        return_actions = [self.actions[i] for i in samples]
        return_rewards = [self.rewards[i] for i in samples]
        return_next_states = [self.next_states[i] for i in samples]
        return_action_spaces = [self.action_spaces[i] for i in samples]
        return_next_action_spaces = [self.next_action_spaces[i] for i in samples]

        return return_states[0], return_actions[0], return_rewards[0], return_next_states[0], return_action_spaces[0], return_next_action_spaces[0]

    def add(self, state, action, reward, next_state, action_space, next_action_space):

        self.current_size = min(self.current_size + 1, self.max_size - 1)

        if self.current_size >= self.max_size:

            self.states.popleft()
            self.actions.popleft()
            self.rewards.popleft()
            self.next_states.popleft()
            self.action_spaces.popleft()
            self.next_action_spaces.popleft()

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.action_spaces.append(action_space)
        self.next_action_spaces.append(next_action_space)

        return
