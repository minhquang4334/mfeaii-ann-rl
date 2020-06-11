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
        print(observation)
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

from ple.games.catcher import Catcher as CatcherEnv
from ple import PLE

class Catcher:

    def __init__(self, width):
        self.dim = 49
        self.game = CatcherEnv(width=width, height=width)
        self.env = PLE(self.game, fps=30, display_screen=False)
        self.width = width
        # self.game.player.vel = vel
        self.allowed_actions = self.env.getActionSet()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def action(self, observation, x):
        # print(observation, observation.shape)
        x = x * 10 - 5
        w1 = x[:32].reshape(4,8)
        b1 = x[32:40]
        w2 = x[40:48].reshape(8,1)
        b2 = x[48]
        out = self.sigmoid(observation @ w1 + b1)
        out = self.sigmoid(out @ w2 + b2)
        ix = int(out > 0.5)
        action = self.allowed_actions[ix]
        return action

    def preprocess(self, observation):
        X = self.width
        Y = self.width       
        state = np.array([
                observation['player_x']/X,
                observation['player_vel']/X,
                observation['fruit_x']/X,
                observation['fruit_y']/X,
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
            if(evaluate >= 500): break
        return - evaluate


from ple.games.pixelcopter import Pixelcopter as PixelcopterEnv
from ple import PLE

class Pixelcopter:

    def __init__(self, momentum):
        self.dim = 73
        self.game = PixelcopterEnv()
        self.env = PLE(self.game, fps=30, display_screen=False)
        self.game.player.momentum = momentum
        self.allowed_actions = self.env.getActionSet()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def action(self, observation, x):
        # print(observation, observation.shape)
        x = x * 10 - 5
        w1 = x[:56].reshape(7,8)
        b1 = x[56:64]
        w2 = x[64:72].reshape(8,1)
        b2 = x[72]
        out = self.sigmoid(observation @ w1 + b1)
        out = self.sigmoid(out @ w2 + b2)
        # print(out.shape)
        ix = int(out > 0.5)
        action = self.allowed_actions[ix]
        return action

    def preprocess(self, observation):
        X = 48.
        Y = 48.       
        state = np.array([
                observation['player_y']/Y,
                observation['player_vel']/X,
                observation['player_dist_to_ceil']/Y,
                observation['player_dist_to_floor']/Y,
                observation['next_gate_dist_to_player']/X,
                observation['next_gate_block_top']/Y,
                observation['next_gate_block_bottom']/Y
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


from ple.games.pong import Pong as PongEnv
from ple import PLE

class Pong:

    def __init__(self, players_speed_ratio):
        self.dim = 73
        self.game = PongEnv(players_speed_ratio=players_speed_ratio)
        self.env = PLE(self.game, fps=30, display_screen=False)
        self.allowed_actions = self.env.getActionSet()

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def action(self, observation, x):
        # print(observation, observation.shape)
        x = x * 10 - 5
        w1 = x[:56].reshape(7,8)
        b1 = x[56:64]
        w2 = x[64:72].reshape(8,1)
        b2 = x[72]
        out = self.sigmoid(observation @ w1 + b1)
        out = self.sigmoid(out @ w2 + b2)
        # print(out.shape)
        ix = int(out > 0.5)
        action = self.allowed_actions[ix]
        return action

    def preprocess(self, observation):
        X = 64.
        Y = 48.       
        state = np.array([
                observation['player_y'],
                observation['player_velocity'],
                observation['cpu_y'],
                observation['ball_x'],
                observation['ball_y'],
                observation['ball_velocity_x'],
                observation['ball_velocity_y']
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
            if(reward > 0): print(reward)
        return - evaluate


def main():
    task = CartPole(9.8)

    y = task.evaluate(np.random.rand(5))
    print(y)

if __name__ == '__main__':
    main()