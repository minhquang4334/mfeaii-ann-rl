import numpy as np

class Sphere:

    def __init__(self, dim):
        self.dim = dim

    def evaluate(self, x):
        return np.sum(np.power(x, 2))

from .environments import CartPoleEnv, AcrobotEnv

class CartPole:

    def __init__(self, gravity):
        self.dim = 5
        self.env = CartPoleEnv()
        self.env.gravity = gravity

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def action(self, observation, x):
        x = x * 10 - 5
        w = x[:4]
        b = x[4]
        return int(self.sigmoid(np.sum(observation * w) + b) > 0.5)

    def evaluate(self, x): # Su dung ket qua best result sau khi hoc trong thuat toan tien hoa de su dung demo
        fitness = 0
        observation = self.env.reset()
        for t in range(200):
            action = self.action(observation, x)
            observation, reward, done, info = self.env.step(action)
            fitness += reward
            if done:
                break
        return - fitness

    def __del__(self):
        self.env.close()

class Acrobot:

    def __init__(self, link_mass_2):
        self.dim = 7
        self.env = AcrobotEnv()
        self.env.LINK_MASS_2 = link_mass_2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def action(self, observation, x):
        x = x * 10 - 5
        w = x[:6]
        b = x[6]
        return int(self.sigmoid(np.sum(observation * w) + b) > 0.5)

    def evaluate(self, x):
        fitness = 0
        observation = self.env.reset()
        for t in range(200):
            action = self.action(observation, x)
            observation, reward, done, info = self.env.step(action)
            fitness += reward
            if done:
                break
        return - fitness

    def __del__(self):
        self.env.close()


from ple.games.flappybird import FlappyBird as FlappyBirdEnv
from ple import PLE

class FlappyBird:

    def __init__(self, gravity):
        self.dim = 9
        self.game = FlappyBirdEnv()
        self.env = PLE(self.game, fps=30, display_screen=False, force_fps=True)
        self.game.player.GRAVITY = gravity
        self.allowed_actions = self.env.getActionSet()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def action(self, observation, x):
        x = x * 10 - 5
        w = x[:8]
        b = x[8]
        ix = int(self.sigmoid(np.sum(observation * w) + b) > 0.5)
        action = self.allowed_actions[ix]
        return action

    def preprocess(self, observation):
        X = 512.
        Y = 288.
        state = np.array([
                observation['player_y'] / Y,
                observation['player_vel'] / X,
                observation['next_pipe_dist_to_player'] / X,
                observation['next_pipe_top_y'] / Y,
                observation['next_pipe_bottom_y'] / Y,
                observation['next_next_pipe_dist_to_player'] / X,
                observation['next_next_pipe_top_y'] / Y,
                observation['next_next_pipe_bottom_y'] / Y,
            ])
        return state

    def evaluate(self, x):
        p = self.env
        fitness = 0
        p.init()
        p.reset_game()
        while 1:
            if p.game_over():
                break
            observation = p.getGameState()
            state = self.preprocess(observation)
            action = self.action(state, x)
            reward = p.act(action)
            # if(reward > 0):
            #     print(reward, self.env.score())
            fitness += reward
        return - fitness

def main():
    task = CartPole(9.8)

    y = task.fitness(np.random.rand(5))
    print(y)

if __name__ == '__main__':
    main()