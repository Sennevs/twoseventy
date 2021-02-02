import matplotlib.pyplot as plt

from games.tictactoe.agent import AI
from games.tictactoe.env import TicTacToeEnv

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], True)



class Simulator:

    def __init__(self, players, env):

        # check if 2 players are playing, only supported mode for now
        if len(players) != 2:
            raise ValueError('Tictactoe simulator is currently only supported for 2 player games.')

        self.players = {player.id: player for player in players}
        self.env = env(self.players.keys())
        self.active_player = None

    def play(self, episodes, greedy=False, max_steps=None):

        games_per_episode = []


        for episode in range(episodes):
            print(f'Episode: {episode}')

            [player.reset() for player in self.players.values()]
            self.active_player = self.env.reset()

            done = False

            import time
            start_time = time.time()

            steps = 0
            while not done:
                active_player = self.players[self.active_player]

                state, previous_rewards, legal_actions = self.env.observe(active_player.id)
                active_player.observe(state, previous_rewards, legal_actions)
                action = active_player.play(state, legal_actions, greedy)

                done, self.active_player, info = self.env.step(self.active_player, action)

                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break

            print(f'Playing player took {time.time() - start_time} seconds')
            start_time = time.time()


            # observe final reward
            games_per_episode.append(steps)
            [player.observe(*self.env.observe(player.id)) for player in self.players.values()]
            [player.update() for player in self.players.values()]

            print(f'Updating player took {time.time() - start_time} seconds')
            start_time = time.time()

        [print(player.reward_hist) for player in self.players.values()]
        return

    def plot_performance(self):

        player_1_rewards = self.players['player_1'].reward_hist
        player_2_rewards = self.players['player_2'].reward_hist

        player_1_turns = self.players['player_1'].turn_hist
        player_2_turns = self.players['player_2'].turn_hist


        print('Player 1 rewards')
        print(sum(player_1_rewards))
        print('Player 2 rewards')
        print(sum(player_2_rewards))

        print(player_1_rewards)
        print(player_2_rewards)

        print('Player 1 turns')
        print(sum(player_1_turns))
        print('Player 2 turns')
        print(sum(player_2_turns))

        print(player_1_turns)
        print(player_2_turns)

        roll_1 = [sum(player_1_rewards[max(0, a - 100):a + 1]) / (a + 1 - max(0, a - 100)) for a, b in
                  enumerate(player_1_rewards)]
        roll_2 = [sum(player_2_rewards[max(0, a - 100):a + 1]) / (a + 1 - max(0, a - 100)) for a, b in
                  enumerate(player_2_rewards)]

        print(roll_1)
        print(roll_2)

        # print(player_1.loss_hist)
        # print(player_2.loss_hist)

        loss_roll_1 = [sum(player_1.loss_hist[max(0, a - 100):a + 1]) / (a + 1 - max(0, a - 100)) for a, b in
                       enumerate(player_1.loss_hist)]
        loss_roll_2 = [sum(player_2.loss_hist[max(0, a - 100):a + 1]) / (a + 1 - max(0, a - 100)) for a, b in
                       enumerate(player_2.loss_hist)]
        plt.plot(list(range(len(loss_roll_1))), loss_roll_1)
        plt.plot(list(range(len(loss_roll_2))), loss_roll_2)
        plt.show()
        loss_roll_1 = loss_roll_1[50:]
        loss_roll_2 = loss_roll_2[50:]

        plt.plot(list(range(len(loss_roll_1))), loss_roll_1)
        plt.plot(list(range(len(loss_roll_2))), loss_roll_2)
        plt.show()

        # plt.scatter(np.asarray(list(range(len(player_1.loss_hist)))), np.asarray(player_1.loss_hist))
        # plt.scatter(np.asarray(list(range(len(player_2.loss_hist)))), np.asarray(player_2.loss_hist))
        # plt.show()

        plt.plot(list(range(len(roll_1))), roll_1)
        plt.plot(list(range(len(roll_2))), roll_2)
        plt.show()

import time

start_time = time.time()
player_1 = AI('player_1')
player_2 = AI('player_2')
simulator = Simulator([player_1, player_2], TicTacToeEnv)
simulator.play(episodes=10000, greedy=False)
simulator.plot_performance()

print(f'Runtime was {time.time() - start_time} seconds.')

simulator.play(episodes=1, greedy=True)
print(simulator.env.visualize_board())

exit()


player_1.save_model()
player_2.save_model()

print('a')
print(list(player_1.replay_buffer.states)[0:10])
print(list(player_1.replay_buffer.rewards)[0:10])
print(list(player_1.replay_buffer.actions)[0:10])
print(list(player_1.replay_buffer.next_states)[0:10])
print(list(player_1.replay_buffer.action_spaces)[0:10])
print(list(player_1.replay_buffer.next_action_spaces)[0:10])
print('b')
print(list(player_2.replay_buffer.states)[0:10])
print(list(player_2.replay_buffer.rewards)[0:10])
print(list(player_2.replay_buffer.actions)[0:10])
print(list(player_2.replay_buffer.next_states)[0:10])
print(list(player_2.replay_buffer.action_spaces)[0:10])
print(list(player_2.replay_buffer.next_action_spaces)[0:10])

play_game(episodes=1, greedy=True, target=True)


