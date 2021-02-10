import tensorflow as tf
import aim
physical_devices = tf.config.list_physical_devices('GPU')

tf.config.experimental.set_memory_growth(physical_devices[0], True)

import matplotlib.pyplot as plt

from games.tictactoe.agent import AI
from games.tictactoe.env import TicTacToeEnv
from games.tictactoe.dql import DQL


class Simulator:

    def __init__(self, players, env, test_players=None, metrics=None):

        # check if 2 players are playing, only supported mode for now
        if len(players) != 2:
            raise ValueError('Tictactoe simulator is currently only supported for 2 player games.')

        self.players = {player.id: player for player in players}
        self.test_players = {player.id: player for player in test_players}
        self.env = env(self.players.keys())
        self.active_player = None

        self.loss_hist = {player.id: [] for player in players}
        self.reward_hist = {player.id: [] for player in players}

        self.metrics = metrics or []


    def play(self, episodes, greedy=False, max_steps=None, update=True, save_model=True, q_lr_player_1=0.001):

        #self.players['player_1'].behavior.q_lr = q_lr_player_1



        #hyperparam_dict = {'player': 'player_1',
        #                   'q_lr': q_lr_player_1}
        #aim.set_params(hyperparam_dict, name='player_1_hyperparams')

        games_per_episode = []

        for episode in range(episodes):

            if episode % 100 == 0:
                print(f'Episode: {episode}')

            [player.reset() for player in self.players.values()]
            self.active_player = self.env.reset()

            done = False

            actions = {player: None for player in self.players}

            #import time
            #start_time = time.time()

            steps = 0
            while not done:
                active_player = self.players[self.active_player]

                state, previous_rewards, legal_actions, done = self.env.observe(active_player.id)
                actions[active_player.id] = active_player.play(state, legal_actions, explore=True)

                #print(f'Player {active_player.id} action:')
                #print(action)

                active_player.observe(state, previous_rewards, legal_actions, done, actions[active_player.id])


                done, self.active_player, info = self.env.step(self.active_player, actions[active_player.id])

                #if update:
                #    self.loss_hist[active_player.id].append(active_player.update())

                steps += 1
                if max_steps is not None and steps >= max_steps:
                    break

            # print(f'Playing player took {time.time() - start_time} seconds')
            # start_time = time.time()

            # observe final reward
            games_per_episode.append(steps)
            [player.observe(*self.env.observe(player.id), actions[player.id]) for player in self.players.values()]
            [self.reward_hist[key].append(value) for key, value in self.env.rewards.items()]
            # print(self.env.rewards)

            if update:
                [self.loss_hist[player.id].append(player.update()) for player in self.players.values()]
                #aim.track(float(self.loss_hist['player_1'][-1].numpy()), name='loss', epoch=episode)

            if episode % 1000 == 0 and episode != 0:
                for key, value in self.loss_hist.items():
                    print(f'{key} Average loss for last 1000 updates: {sum(value[-1000:]) / len(value[-1000:])}')
                for key, value in self.reward_hist.items():
                    print(f'{key} Average reward for last 1000 updates: {sum(value[-1000:]) / len(value[-1000:])}')
                for key, value in self.reward_hist.items():
                    print(f'{key} Average number of ties for last 1000 updates: {value[-1000:].count(0) / len(value[-1000:])}')

            if episode % 1000 == 0 and episode != 0:
                print('Visualizing board:')
                print(self.env.visualize_board())

                self.test()

            # print(f'Updating player took {time.time() - start_time} seconds')

        # [print(player.reward_hist) for player in self.players.values()]

        if save_model:
            [player.save_model() for player in self.players.values()]
        return

    def test(self):

        score = float(0)
        losses = 0
        test_env = TicTacToeEnv(self.test_players.keys())
        for episode in range(1000):

            [player.reset() for player in self.test_players.values()]
            self.active_player = test_env.reset()

            done = False
            action_hist = []

            #print(f'Game {episode}')

            steps = 0
            while not done:
                active_player = self.test_players[self.active_player]

                state, previous_rewards, legal_actions, done = test_env.observe(active_player.id)
                action = active_player.play(state, legal_actions, explore=False, target=False)
                action_hist.append(action)
                #print(f'Player {active_player.id}')
                #print(action)
                #print(action)
                done, self.active_player, info = test_env.step(self.active_player, action)

                steps += 1

            # observe final reward
            score += test_env.rewards['player_1']
            losses += 1 if test_env.rewards['player_1'] == -1 else 0

            if test_env.rewards['player_1'] == -1:
                print(f'Episode: {episode}')
                for a in action_hist:
                    print(a)


            #print(test_env.rewards['player_1'])

        print(f'Score for player 1: {score}')
        print(f'Losses for player 1: {losses}')


        return score

    def plot_performance(self):

        player_1_rewards = self.reward_hist['player_1']
        player_2_rewards = self.reward_hist['player_2']

        player_1_loss = self.loss_hist['player_1']
        player_2_loss = self.loss_hist['player_2']

        #player_1_turns = self.players['player_1'].turn_hist
        #player_2_turns = self.players['player_2'].turn_hist


        #print('Player 1 rewards')
        #print(sum(player_1_rewards))
        #print('Player 2 rewards')
        #print(sum(player_2_rewards))

        #print(player_1_rewards)
        #print(player_2_rewards)

        #print('Player 1 turns')
        #print(sum(player_1_turns))
        #print('Player 2 turns')
        #print(sum(player_2_turns))

        #print(player_1_turns)
        #print(player_2_turns)

        roll_1 = [sum(player_1_rewards[max(0, a - 100):a + 1]) / (a + 1 - max(0, a - 100)) for a, b in
                  enumerate(player_1_rewards)]
        roll_2 = [sum(player_2_rewards[max(0, a - 100):a + 1]) / (a + 1 - max(0, a - 100)) for a, b in
                  enumerate(player_2_rewards)]

        #print(roll_1)
        #print(roll_2)

        # print(player_1.loss_hist)
        # print(player_2.loss_hist)

        loss_roll_1 = [sum(player_1_loss[max(0, a - 100):a + 1]) / (a + 1 - max(0, a - 100)) for a, b in
                       enumerate(player_1_loss)]
        loss_roll_2 = [sum(player_2_loss[max(0, a - 100):a + 1]) / (a + 1 - max(0, a - 100)) for a, b in
                       enumerate(player_2_loss)]
        plt.plot(list(range(len(loss_roll_1))), loss_roll_1)
        plt.plot(list(range(len(loss_roll_2))), loss_roll_2)
        plt.show()
        loss_roll_1 = loss_roll_1[1000:]
        loss_roll_2 = loss_roll_2[1000:]

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
behavior_1 = DQL(action_space=9)
behavior_2 = DQL(action_space=9)
behavior_3 = DQL(action_space=9, random=True)
player_1 = AI('player_1', behavior=behavior_1)
player_2 = AI('player_2', behavior=behavior_2)
player_random = AI('player_3', behavior=behavior_3)
simulator = Simulator([player_1, player_2], TicTacToeEnv, [player_1, player_random])

#player_1.load_model()
#player_2.load_model()

#simulator.play(episodes=1, greedy=True, update=False, save_model=False)
#print(simulator.env.visualize_board())


#exit()

simulator.play(episodes=50000, greedy=False)
simulator.plot_performance()
#sess.close()

print(f'Runtime was {time.time() - start_time} seconds.')

simulator2 = Simulator([player_1, player_random], TicTacToeEnv)
simulator2.play(episodes=100, greedy=True)
print(simulator.env.visualize_board())

exit()

experiments = [0.1, 0.01, 0.001]

for experiment in experiments:
    sess = aim.Session(run=f'{int(1000000*experiment)}_b')
    simulator.play(episodes=10, greedy=False, q_lr_player_1=experiment)
    simulator.plot_performance()
    sess.close()

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


