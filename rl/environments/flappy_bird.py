from ple.games.flappybird import FlappyBird
from ple import PLE

import numpy as np
import random

class myAgentHere:

    def __init__(self, allowed_actions):
        self.allowed_actions = allowed_actions

    def pickAction(self, reward, observation):
        return np.random.choice(self.allowed_actions)

game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)
agent = myAgentHere(allowed_actions=p.getActionSet())

p.init()
reward = 0.0
nb_frames = 1000

for i in range(nb_frames):
   observation = p.getScreenRGB()
   print(observation.shape)
   action = agent.pickAction(reward, observation)
   reward = p.act(action)
   if p.game_over():
        break
