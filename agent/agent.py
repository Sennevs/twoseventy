from .q_network import QFunction

from ..env import Env


class Agent:

    def __init__(self, id):

        self.q_network = QFunction()
        self.id = id
        self.player = None

        return

    def train(self, episodes, max_steps, ):

        # get state and legal actions
        env = Env()

        for i_episode in range(20):
            state = env.reset()
            for t in range(100):
                print(state)
                action = env.action_space.sample()
                observation, reward, done, info = env.step(action)
                if done:
                    print("Episode finished after {} timesteps".format(t + 1))
                    break
        env.close()

        # calculate q-values for legal actions

        # execute legal action

        # observe

        return

    def play(self, env):

        playing = False
        while playing:
            # get legal actions from simulator
            legal_actions = env.get_legal_actions(self.id)

            # loop over legal actions and select action with highest q-value\
            optimal_action = Agent.predict(state, legal_actions)

            # input optimal action in env, get state back and playing variable, playing variable is always true except if input is end_turn
            state, playing = env.play_action(optimal_action, self.id)

        return

    def choose_player(self):

        self.player = {'player': self.id, 'candidate': 'BDR'}

        return self.player

    def predict(self, state, legal_actions):

        q_values = []
        highest_q = -100000000
        opt_action =

        for action in legal_actions:
            qv = self.q_network([state, action])

            if qv > highest_q:

                highest_q = qv
                #opt_action =



        return


import argparse

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--players', action='store', type=str, nargs='*')
    parser.add_argument('--ui', type=str, required=False)
    args = vars(parser.parse_args())

    if args['players'] is None:
        args['players'] = [{'name': 'Senne', 'candidate': 'BD', 'type': 'ai'},
                           {'name': 'Barry', 'candidate': 'MPR', 'type': 'human'}]

    sim = Env(args['players'])
    sim.play(args['ui'])

    return


if __name__ == "__main__":
    main()
