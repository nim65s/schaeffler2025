# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import random

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
import matplotlib.pylab as plt
import unittest

# HYPERPARAM
SEED = 1
ENV_ID = "CartPole-v1"
NUM_ENVS = 1
LEARNING_RATE = 0.00025
BUFFER_SIZE = 10000
TOTAL_TIMESTEPS = 500000
TRAIN_FREQUENCY = 10
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_NETWORK_FREQUENCY = 500
TAU = 1.0

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

# TRY NOT TO MODIFY: seeding
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# env setup
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk

envs = gym.vector.SyncVectorEnv(
    [make_env(ENV_ID, SEED + i, i, False, "TP6") for i in range(NUM_ENVS)]
)
assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

q_network = QNetwork(envs)
optimizer = torch.optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
target_network = QNetwork(envs)
target_network.load_state_dict(q_network.state_dict())

rb = ReplayBuffer(
    BUFFER_SIZE,
    envs.single_observation_space,
    envs.single_action_space,
    "cpu",
    handle_timeout_termination=False,
)

# start the game
obs, _ = envs.reset(seed=SEED)
started = np.zeros(envs.num_envs, dtype=bool)
h_rwd = []

for global_step in range(TOTAL_TIMESTEPS):
    # ALGO LOGIC: put action logic here
    epsilon = linear_schedule(1, 0.05, .5 * TOTAL_TIMESTEPS, global_step)
    if random.random() < epsilon:
        actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    else:
        q_values = q_network(torch.Tensor(obs))
        actions = torch.argmax(q_values, dim=1).numpy()

    # execute the game and log data.
    next_obs, rewards, terminations, truncations, infos = envs.step(actions)

    # record rewards for plotting purposes
    if "episode" in infos:
        for idx in range(NUM_ENVS):
            if truncations[idx] or terminations[idx]:
                assert infos["episode"]["l"]>0
                h_rwd.append(infos["episode"]["r"][idx])
                if len(h_rwd) % 100 == 0:  # verbose one every ~200 episode
                    print(f"global_step={global_step}/{TOTAL_TIMESTEPS},"
                          + f" episodic_return={h_rwd[-1]}")

    # Add samples to replay buffer if env not starting
    if np.any(started):
        rb.add(obs[started], next_obs[started], actions[started],
                rewards[started], terminations[started], infos)
    obs = next_obs
    started = np.logical_not(np.logical_or(terminations, truncations))

    # CRUCIAL step easy to overlook
    obs = next_obs

    # ALGO LOGIC: training.
    if global_step > TOTAL_TIMESTEPS // 50 and global_step % TRAIN_FREQUENCY == 0:
        data = rb.sample(BATCH_SIZE)
        with torch.no_grad():
            target_max, _ = target_network(data.next_observations).max(dim=1)
            td_target = data.rewards.flatten() + GAMMA * target_max * (1 - data.dones.flatten())
        old_val = q_network(data.observations).gather(1, data.actions).squeeze()
        loss = F.mse_loss(td_target, old_val)

        if int(round((global_step/100) ** (1/3))) ** 3 ==  global_step/100:
            print(f'*** Learning goes ... loss = {loss},  q = {old_val.mean()}')

        # optimize the model
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # update target network
    if global_step % TARGET_NETWORK_FREQUENCY == 0:
        for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
            target_network_param.data.copy_(
                TAU * q_network_param.data + (1.0 - TAU) * target_network_param.data
            )

envs.close()

# ### DISPLAY RESULTS
# ### Plot the learning curve
print(f"Total rate of success: {np.mean(h_rwd)}")
plt.plot(np.cumsum(h_rwd) / range(1, len(h_rwd)+1))

def rendertrial(env):
    """Roll-out from random state using greedy policy."""
    s,_ = env.reset()
    traj = [s]
    while True:
        a = int(torch.argmax(q_network(torch.Tensor(s).unsqueeze(0))))
        s, r, done, trunc, _ = env.step(a)
        traj.append(s)
        if done or trunc: break
    return traj

envrender = gym.make('CartPole-v1',render_mode = 'human')
traj = rendertrial(envrender)


hq = []
hp = []
ha = []
for pos in np.arange(envs.single_observation_space.low[0],
                     envs.single_observation_space.high[0],.1):
    for angle in np.arange(envs.single_observation_space.low[2],
                           envs.single_observation_space.high[2],.01):
        hp.append(pos)
        ha.append(angle)
        hq.append(float(torch.max(
            q_network(torch.tensor([[pos, 0, angle, 0]], dtype=torch.float32)))))

# Scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(hp, ha, c=hq, cmap='viridis')
plt.colorbar(label='Max Q-value')
plt.xlabel('Cart Position')
plt.ylabel('Pole Angle')
plt.title('Q-value landscape (velocity = 0)')
print('Type plt.show() to display the result')

### TEST ZONE ############################################################
### This last part is to automatically validate the versions of this example.
class MyTest(unittest.TestCase):
    def test_conv(self):
        self.assertGreater(np.average(h_rwd[-20:]), -15)

MyTest().test_conv()
