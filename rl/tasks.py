import numpy as np

class Sphere:

    def __init__(self, dim):
        self.dim = dim

    def evaluate(self, x):
        return np.sum(np.power(x, 2))

from .environments import CartPoleEnv

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

    def evaluate(self, x):
        evaluate = 0
        observation = self.env.reset()
        for t in range(200):
            action = self.action(observation, x)
            observation, reward, done, info = self.env.step(action)
            evaluate += reward
            if done:
                break
        return - evaluate

    def __del__(self):
        self.env.close()

from .environments import AcrobotEnv

class Acrobot:

    # def __init__(self, link_mass_2):
    #     self.dim = 7
    #     self.env = AcrobotEnv()
    #     self.env.LINK_MASS_2 = link_mass_2

    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    # def action(self, observation, x):
    #     x = x * 10 - 5
    #     w = x[:6]
    #     b = x[6]
    #     return int(self.sigmoid(np.sum(observation * w) + b) > 0.5)
    def __init__(self, link_mass_2):
        self.dim = 65
        self.env = AcrobotEnv()
        self.env.LINK_MASS_2 = link_mass_2

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def action(self, observation, x):
        x = x * 10 - 5
        w1 = x[:48].reshape(6,8)
        b1 = x[48:56]
        w2 = x[56:64].reshape(8,1)
        b2 = x[64]
        out = self.sigmoid(observation @ w1 + b1)
        out = self.sigmoid(out @ w2 + b2)
        return int(out > 0.5)

    def evaluate(self, x):
        evaluate = 0
        observation = self.env.reset()
        for t in range(200):
            action = self.action(observation, x)
            observation, reward, done, info = self.env.step(action)
            evaluate += reward
            if done:
                break
        return - evaluate

    def __del__(self):
        self.env.close()


from ple.games.flappybird import FlappyBird as FlappyBirdEnv
from ple import PLE
import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"
class FlappyBird:

    # def __init__(self, gravity):
    #     self.dim = 9
    #     self.game = FlappyBirdEnv()
    #     self.env = PLE(self.game, fps=30, display_screen=False)
    #     self.game.player.GRAVITY = gravity
    #     self.allowed_actions = self.env.getActionSet()

    # def sigmoid(self, x):
    #     return 1 / (1 + np.exp(-x))

    # def action(self, observation, x):
    #     x = x * 10 - 5
    #     w = x[:8]
    #     b = x[8]
    #     ix = int(self.sigmoid(np.sum(observation * w) + b) > 0.5)
    #     action = self.allowed_actions[ix]
    #     return action

    def __init__(self, gravity):
        self.dim = 61
        self.game = FlappyBirdEnv()
        self.env = PLE(self.game, fps=30, display_screen=False)
        self.game.player.GRAVITY = gravity
        self.allowed_actions = self.env.getActionSet()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def action(self, observation, x):
        x = x * 10 - 5
        w1 = x[:32].reshape(8,4)
        b1 = x[32:36]
        w2 = x[36:52].reshape(4,4)
        b2 = x[52:56]
        w3 = x[56:60].reshape(4,1)
        b3 = x[60]
        out = self.sigmoid(observation @ w1 + b1)
        out = self.sigmoid(out @ w2 + b2)
        out = self.sigmoid(out @ w3 + b3)

        ix = int(out > 0.5)
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
        evaluate = 0
        p.init()
        p.reset_game()
        while 1:
            if p.game_over():
                break
            observation = p.getGameState()
            state = self.preprocess(observation)
            action = self.action(state, x)
            reward = p.act(action)
            evaluate += reward
        return - evaluate


from .environments import MountainCarEnv

class MoutainCar:

    def __init__(self, gravity):
        self.dim = 27
        self.env = MountainCarEnv()
        self.env.gravity = gravity

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def action(self, observation, x):
        x = x * 10 - 5
        w1 = x[:8].reshape(2,4)
        b1 = x[8:12]
        w2 = x[12:24].reshape(4,3)
        b2 = x[24:27]
        # print (observation.shape, w2.shape, b2.shape)
        out = self.sigmoid(observation @ w1 + b1)
        out = self.sigmoid(out @ w2 + b2)
        idx = np.argmax(out)
        return idx

    def evaluate(self, x):
        evaluate = 0
        observation = self.env.reset()
        for t in range(200):
            action = self.action(observation, x)
            observation, reward, done, info = self.env.step(action)
            evaluate += reward
            if done:
                break
        # print(self.env.state)
        return - evaluate

    def __del__(self):
        self.env.close()

def main():
    task = CartPole(9.8)

    y = task.evaluate(np.random.rand(5))
    print(y)

if __name__ == '__main__':
    main()