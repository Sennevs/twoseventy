import numpy as np

from games import Env


class Player:

    def __init__(self, name, player_id):

        self.observed_state = None
        self.name = name
        self.id = player_id

        return


class TicTacToeEnv(Env):

    def __init__(self, players):

        if len(players) != 2:
            raise ValueError('Currently only 2 players is supported.')

        self.players = [Player(player, player_id=i) for i, player in enumerate(players)]
        self.board_size = 3
        self.board = np.zeros((self.board_size, self.board_size))
        for player in self.players:
            player.observed_state = self.board
        self.active_player = 0
        self.inactive_player = 1
        self.info = {}

        self.name_to_idx = {player.name: idx for idx, player in enumerate(self.players)}

        self.done = False
        self.winner = None


        return

    def get_legal_actions(self):

        return np.absolute(self.board).reshape([9, 1])

    def observe(self, player):

        player = self.name_to_idx[player]

        legal_actions = self.get_legal_actions()

        return self.players[player].observed_state.reshape([1, 9]), legal_actions

    def check_win(self):

        eval_board = self.players[self.active_player].observed_state == 1

        rows = eval_board.sum(axis=0)
        cols = eval_board.sum(axis=1)
        diag = np.fliplr(eval_board).diagonal().sum(axis=0).reshape([1])
        adiag = eval_board.diagonal().sum(axis=0).reshape([1])

        res = np.concatenate([rows, cols, diag, adiag], axis=0)

        if 3 in res:
            self.winner = self.players[self.active_player].name
            self.done = True

        if not np.any(self.board == 0):
            self.done = True
            self.winner = 'tie'

        return self.winner

    def update_player_turn(self):

        if self.active_player == 0:
            self.active_player = 1
            self.inactive_player = 0
        else:
            self.active_player = 0
            self.inactive_player = 1

        return

    def confirm_move(self, action):

        new_board = self.board + action

        changes = 0
        legal = True
        for old_field, new_field in zip(self.board.flatten(), new_board.flatten()):
            if old_field != new_field:
                changes += 1
                if old_field != 0 or (new_field not in (1, -1)):
                    legal = False

        if (not legal) or (changes != 1):
            print('State:')
            print(state)
            print('Action:')
            print(action)
            raise ValueError('Move was illegal and has not been confirmed.')

        self.board = new_board

        self.players[0].observed_state = self.board
        self.players[1].observed_state = -self.board

        return

    def _standardize_action(self, action):

        if self.active_player == 1:
            action *= -1
        action = action.reshape([3, 3])

        return action

    def _update_rewards(self):

        if self.winner == 'tie':
            rewards = {player.name: 0 for player in self.players}
        else:
            rewards = {player.name: int(player.name == self.winner) for player in self.players}
            if self.done:
                rewards = {key: -1 if (value == 0) else 1 for key, value in rewards.items()}

        return rewards


    def step(self, action):

        # standardize action
        action = self._standardize_action(action)

        # update game state
        self.confirm_move(action)

        # check if there's a winner
        self.check_win()

        # update active player
        self.update_player_turn()

        rewards = self._update_rewards()

        return rewards, self.done, self.players[self.active_player].name, self.info

    def reset(self):

        self.board_size = 3
        self.board = np.zeros((self.board_size, self.board_size))
        for player in self.players:
            player.observed_state = self.board
        self.active_player = 0
        self.info = {}

        self.done = False

        return self.players[self.active_player].observed_state, self.players[self.active_player].name

    def close(self):

        return

from games.tictactoe.agent import AI

player_1 = AI(1)
player_2 = AI(2)
env = TicTacToeEnv(['player_1', 'player_2'])


def play_game(episodes=100, greedy=False, target=False, plot=False):
    num_episodes = episodes

    player_1_rewards = []
    player_2_rewards = []
    for episode in range(num_episodes):

        print(f'Episode: {episode}')

        player_1.reset()
        player_2.reset()
        state, active_player = env.reset()
        done_z = False
        rewards_z = None
        first_1 = True
        first_2 = True


        step = 0
        while not done_z:

            if active_player == 'player_1':
                state_1, legal_actions_1 = env.observe('player_1')
                action_1 = player_1.play(state_1, legal_actions_1, greedy=greedy, target=target)

                #print(f'Action: {action_1}')
                rewards_z, done_z, active_player, info = env.step(action_1)

            else:
                state_2, legal_actions_2 = env.observe('player_2')
                action_2 = player_2.play(state_2, legal_actions_2, greedy=greedy, target=target)

                #print(f'Action: {action_2}')
                rewards_z, done_z, active_player, info = env.step(action_2)
            #print(f'Done: {done_z}')

            player_1.update()
            player_2.update()

            #print(active_player)

            if not first_1:
                if active_player == 'player_1':
                    player_1.observe(state_1, action_1, rewards_z['player_1'], legal_actions_1)
            if not first_2:
                if active_player == 'player_2':
                    player_2.observe(state_2, action_2, rewards_z['player_2'], legal_actions_2)

            if done_z:
                #player_1.observe(state_1, action_1, rewards_z['player_1'], legal_actions_1)
                #player_2.observe(state_2, action_2, rewards_z['player_2'], legal_actions_2)

                state_1, legal_actions_1 = env.observe('player_1')
                player_1.observe(state_1, action_1, 0, None)
                state_2, legal_actions_2 = env.observe('player_2')
                player_2.observe(state_2, action_2, 0, None)


            #print(f'Reward: {rewards_z}')

            step += 1

            if active_player == 'player_1':
                first_2 = False
            else:
                first_1 = False

        player_1_rewards.append(rewards_z['player_1'])
        player_2_rewards.append(rewards_z['player_2'])

        print(f'Reward: {rewards_z}')
        print('Final board:')
        print(env.board)




    print('Player 1 rewards')
    print(sum(player_1_rewards))
    print('Player 2 rewards')
    print(sum(player_2_rewards))

    print(player_1_rewards)
    print(player_2_rewards)


    roll_1 = [sum(player_1_rewards[max(0, a-100):a + 1])/(a + 1 - max(0, a-100)) for a, b in enumerate(player_1_rewards)]
    roll_2 = [sum(player_2_rewards[max(0, a-100):a + 1])/(a + 1 - max(0, a-100)) for a, b in enumerate(player_2_rewards)]

    print(roll_1)
    print(roll_2)

    import matplotlib.pyplot as plt

    #print(player_1.loss_hist)
    #print(player_2.loss_hist)

    if plot:
        loss_roll_1 = [sum(player_1.loss_hist[max(0, a-100):a + 1])/(a + 1 - max(0, a-100)) for a, b in enumerate(player_1.loss_hist)]
        loss_roll_2 = [sum(player_2.loss_hist[max(0, a-100):a + 1])/(a + 1 - max(0, a-100)) for a, b in enumerate(player_2.loss_hist)]
        plt.plot(list(range(len(loss_roll_1))), loss_roll_1)
        plt.plot(list(range(len(loss_roll_2))), loss_roll_2)
        plt.show()
        loss_roll_1 = loss_roll_1[50:]
        loss_roll_2 = loss_roll_2[50:]
        plt.plot(list(range(len(loss_roll_1))), loss_roll_1)
        plt.plot(list(range(len(loss_roll_2))), loss_roll_2)
        plt.show()

        #plt.scatter(np.asarray(list(range(len(player_1.loss_hist)))), np.asarray(player_1.loss_hist))
        #plt.scatter(np.asarray(list(range(len(player_2.loss_hist)))), np.asarray(player_2.loss_hist))
        #plt.show()

        plt.plot(list(range(len(roll_1))), roll_1)
        plt.plot(list(range(len(roll_2))), roll_2)
        plt.show()


play_game(episodes=10000, target=False, plot=True)
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
exit()

from games.tictactoe.agent import AI

state_space = [9, 1]
action_space = [9, 1]

player_1 = AI(1)
player_2 = AI(2)
env = TicTacToeEnv(['player_1', 'player_2'])
state, active_player = env.reset()
done = False
reward_z = None
while not done:


    x = input(f'Player {active_player} make your move: ')
    action_z = np.zeros((9,1))
    action_z[int(x)] = 1

    print(f'Action: {action_z}')
    rewards_z, done, active_player, info = env.step(action_z)



    print(env.observe(active_player))

print(rewards_z)



