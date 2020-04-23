import gym
from cartpole import CartPoleEnv

env = CartPoleEnv()

observation = env.reset()

total_reward = 0

print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

for t in range(200):
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    total_reward += reward
    if done:
        break

env.close()

print(total_reward)