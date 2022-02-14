import torch
from matplotlib import pyplot as plt
import numpy as np
from tqdm.auto import tqdm, trange
from pathlib import Path

from rcmodel import *

"""
This script trains a policy using a pre defined prior policy.
Rewards are then the difference between the outputs of both policies.
The trained policy can be saved and trained further using data.

A sin wave is used to create dummy room temperature data and the responses of the prior and policy are compared.
"""

actions = []
hot_rm = []

day = 24*60**2

pi = PolicyNetwork(5, 2)
rm_CA = [100, 1e4]  # [min, max] Capacitance/area
ex_C = [1e3, 1e8]  # Capacitance
R = [0.1, 5]  # Resistance ((K.m^2)/W)
Q_limit = [-100, 100]  # Cooling limit and gain limit in W/m2
scaling = InputScaling(rm_CA, ex_C, R, Q_limit)

# weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather ' \
                        'Files/JuneSept.csv'

model = initialise_model(pi, scaling, weather_data_path)


time_data = torch.arange(0, 30 * day, 30)
temp_data = 3.5*torch.sin(time_data * (2 * np.pi/day) - 2/3*day) + 22.5  # dummy rm temperature data.
temp_data = temp_data.unsqueeze(0).T
env = PriorEnv(model, time_data, temp_data, PriorCoolingPolicy())

rl = Reinforce(env)

rewards_plot = []
ER_plot = []
opt_id = 0
Path(f'./outputs/run{opt_id}/plots/results/').mkdir(parents=True, exist_ok=True)

epochs = 3000

for epoch in trange(epochs):

    # Policy Training:
    num_episodes = 1
    step_size = day
    rewards, ER = rl.train(num_episodes, step_size)
    rewards_plot.append(rewards)
    ER_plot.append(ER)

    tqdm.write(f'Epoch {epoch + 1}, Policy Rewards {sum(rewards).item():.1f}, Policy Expected Rewards {sum(ER).item():.1f}')

fig, axs = plt.subplots(1, 2, figsize=(10, 7), dpi=400)
axs[0].plot(range(1, epochs + 1), torch.flatten(torch.tensor(ER_plot)).detach().numpy(), 'b',
            label='expected rewards')
axs[0].legend()

axs[1].plot(range(1, epochs + 1), torch.flatten(torch.tensor(rewards_plot)).detach().numpy(), 'r',
            label='total rewards')
axs[1].legend()
fig.suptitle('Rewards', fontsize=16)
axs[0].set_xlabel('Epoch')
axs[1].set_xlabel('Epoch')
axs[0].set_ylabel('Reward')
axs[1].set_ylabel('Reward')

plt.savefig(f'./outputs/run{opt_id}/plots/RewardsPlot.png')
plt.show()
# plt.close()


model.save()
