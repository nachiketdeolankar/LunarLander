# Importing the Dependencies
import gym
from stable_baselines import ACER 
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.evaluation import evaluate_policy
from matplotlib import animation
import matplotlib.pyplot as plt

# OpenAIGym
environment_name = 'LunarLander-v2'

"""
# Random Actions on Environment
env = gym.make(environment_name)
episodes = 10
for episode in range(0, episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward

    print('Episode:{} Score:{}'.format(episode, score))
env.close()
"""

# Building the model
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
model = ACER('MlpPolicy', env, verbose = 1)

# Training the model
model.learn(total_timesteps = 100000)

# Save and Test The Model
evaluate_policy(model, env, n_eval_episodes = 10, render = True)
env.close()

model.save('ACER_Lander')

# Save the render as gif
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

# Loading the model
del model
model = ACER.load("ACER_Lander", env = env)
obs = env.reset()
frames = []
while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    frames.append(env.render(mode="rgb_array"))

#save_frames_as_gif(frames)