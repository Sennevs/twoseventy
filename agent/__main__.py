from .agent import Agent
from ..env import Env

NUMBER_OF_AGENTS = 2

max_episode_steps = 20
max_turn_steps = 20
episodes = 100

# create agents
agents = [Agent(str(id)) for id in range(0, NUMBER_OF_AGENTS)]

# here we need to automate choosing the optimal candidate, for now, this is hardcoded in Agent
agent_players = [agent.choose_player() for agent in agents]

env = Env(agent_players)

for episode in range(0, 100):

    states = env.reset()
    
    done = False
    step = 0
    while not done:

        actions = []
        for agent, state in (agents, states):
            actions.append(agent.play(max_turn_steps))

        env.step(actions)

        step += 1
        if step >= max_episode_steps:
            break







