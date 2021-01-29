from .agent import Agent
from games.twoseventy import Env

from games.tictactoe import TicTacToeEnv

NUMBER_OF_AGENTS = 2

max_episode_steps = 20
max_turn_steps = 20
episodes = 100

# create agents
agents = {f'agent_{agent_id}': Agent(str(id)) for agent_id in range(0, NUMBER_OF_AGENTS)}

# here we need to automate choosing the optimal candidate, for now, this is hardcoded in Agent
# move this part to twoseventy
# agent_players = [agent.choose_player() for agent in agents]
# env = Env(agent_players)

env = TicTacToeEnv(agents.keys())
for episode in range(0, 100):

    state, current_player = env.reset()
    
    done = False
    step = 0
    while not done:

        action = agents[current_player].action()

        state, reward, done, next_player, info = env.step(action)

        step += 1
        if step >= max_episode_steps:
            break







