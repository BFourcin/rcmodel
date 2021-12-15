#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path

from torch import Tensor
from tqdm.auto import tqdm, trange

from rcmodel import *
from main import initialise_model, RayActor


# In[2]:


torch.cuda.is_available = lambda: False


# In[3]:


import gym
from gym import spaces

class LSIEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, RC, time_data):
        super().__init__()

        self.RC = RC  # RCModel Class
        self.time_data = time_data  # timeseries
        self.t_index = 0  # used to keep track of index through timeseries

        # ----- GYM Stuff -----
        self.n_rooms = len(self.RC.building.rooms)
        self.low_state = -10
        self.high_state = 50
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2, )

        # Observation is temperature of each room.
        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            shape=(self.n_rooms,),
            dtype=np.float32
        )

#                  action
    def step(self, step_size):
        # Execute a chunk of timeseries
        t_eval = self.time_data[self.t_index:int(self.t_index + step_size)]

        # actions are decided and stored by the policy while integrating the ODE:
        pred = self.RC(t_eval)
        
        return pred.detach()  # No need for grad
    #          (observation, reward, done, info)
    # self.state, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.t_index = 0
        self.RC.reset_iv()

        self.RC.cooling_policy.on_policy_reset()

    def render(self, mode='human', close=False):
        # Render the environment to the screen

        return


# In[4]:




opt_id = 0

weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/test2d_sorted.csv'
#
# weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
# csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'  # where building data
dir_path = f'./outputs/run{opt_id}/models/'  # where to save
# Load model?
load_model_path = None  # or None
start_num = 0  # Number of cycles to start at. Used if resuming the run. i.e. the first cycle is (start_num + 1)



def do_plots(ER_plot, rewards_plot, opt_id):
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 7), dpi=400)
    y = torch.flatten(torch.tensor(ER_plot)).detach().numpy()
    axs[0].plot(range(1, len(y) + 1), y, 'b', label='expected rewards')
    axs[0].legend()

    y = torch.flatten(torch.tensor(rewards_plot)).detach().numpy()
    axs[1].plot(range(1, len(y) + 1), y, 'r', label='total rewards')
    axs[1].legend()
    fig.suptitle('Rewards', fontsize=16)
    axs[0].set_xlabel('Epoch')
    axs[1].set_xlabel('Epoch')
    axs[0].set_ylabel('Reward')
    axs[1].set_ylabel('Reward')

    fig.savefig(f'./outputs/run{opt_id}/plots/RewardsPlot.png')
    plt.close()


def init_scaling():
    # Initialise scaling class
    rm_CA = [100, 1e4]  # [min, max] Capacitance/area
    ex_C = [1e3, 1e8]  # Capacitance
    R = [0.1, 5]  # Resistance ((K.m^2)/W)
    scaling = InputScaling(rm_CA, ex_C, R)
    return scaling


# Initialise Model
scaling = init_scaling()

# Initialise Optimise Class - for training physical model
dt = 30
sample_size = 24 * 60 ** 2 / dt

# Initialise Reinforcement Class - for training policy
time_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 1], dtype=torch.float64)
temp_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32), dtype=torch.float32)


# Dataloader to be used when plotting a model run. This is just for info.
plot_results_data = tools.BuildingTemperatureDataset(csv_path, 5 * sample_size, all=True)
plot_dataloader = torch.utils.data.DataLoader(plot_results_data, batch_size=1, shuffle=False)






# In[ ]:





# In[16]:


def train_policy(rl, opt_id):
    # lists to keep track of process, used for plot at the end
    avg_train_loss_plot = []
    avg_test_loss_plot = []
    rewards_plot = []
    ER_plot = []
    count = 1  # Counts number of epochs since start.


    # check if dir exists and make if needed
    Path(f'./outputs/run{opt_id}/plots/results/').mkdir(parents=True, exist_ok=True)

    # Convergence happens when the mean gradient of loss/reward is < tol.
    tol_policy = 50


    cycles = 1
    max_epochs = 200

    plt.ioff()  # Reduces memory usage by matplotlib
    for cycle in range(cycles):
        # -------- Policy training --------
        tqdm.write('Policy Training:')
        diff = torch.ones(5) * 1000
        rewards_prev = 0
        for epoch in trange(max_epochs):
            rewards, ER = rl.train(1, sample_size)

            indx = epoch % len(diff)
            diff[indx] = abs(rewards_prev - torch.tensor(rewards))  # keeps track of a window of differences
            rewards_prev = torch.tensor(rewards)
            rewards_plot.append(rewards)
            ER_plot.append(ER)

            tqdm.write(
                f'Epoch {count}, Rewards/Expected Rewards: {torch.tensor(rewards).sum().item():.2f}/{torch.tensor(ER).sum().item():.2f}, Mean diff: {torch.mean(diff):.2f}')

            # Save Model
            model_id = count + start_num
            rl.env.RC.save(model_id, dir_path)
            count += 1

#             if torch.mean(diff) < tol_policy:
#                 break

        tqdm.write(f'Policy converged in {epoch + 1} epochs. Total epochs: {count - 1}\n')

        # Save a plot of results after policy training
        pltsolution_1rm(rl.env.RC, plot_dataloader, f'./outputs/run{opt_id}/plots/results/Result_Cycle{start_num + cycle + 1}b.png')

        # Plot loss and reward plot
        do_plots(ER_plot, rewards_plot, opt_id)


# In[ ]:





# In[6]:


class Reinforce_baseline:
    def __init__(self, env, time_data, temp_data, gamma=0.9999, alpha=1e-3):

        assert len(time_data) == len(temp_data)

        self.env = env
        # self.pi = pi
        self.time_data = time_data
        self.temp_data = temp_data
        self.gamma = gamma
        self.alpha = alpha
        self.baseline = 0

        self.optimiser = torch.optim.Adam(self.env.RC.cooling_policy.parameters(), lr=self.alpha)

    def update_policy(self, rewards, log_probs):

        # Calculate Discounted Reward:
        discounted_rewards = torch.zeros(len(rewards))

        R = 0
        indx = len(rewards) - 1
        for r in reversed(rewards):
            R = r + self.gamma * R  # Discounted Reward is calculated from last reward to first.

            discounted_rewards[indx] = R  # Fill array back to front to un-reverse the order
            indx -= 1

        expected_reward = -torch.stack(log_probs) * (discounted_rewards - self.baseline)  # negative for maximising
        expected_reward = torch.sum(expected_reward)

        # Update parameters in pi
        self.optimiser.zero_grad()
        expected_reward.backward()
        self.optimiser.step()

        # print(list(self.pi.parameters())[0].grad)  # check on grads if needed

        return expected_reward, discounted_rewards

    def train(self, num_episodes, step_size):
        self.env.RC.cooling_policy.train()  # Put in training mode
        total_ER = []
        total_rewards = []

        loss_fn = torch.nn.MSELoss(reduction='none')  # Squared Error

        # with tqdm(total=len(self.env.time_data) * num_episodes, position=0, leave=False) as pbar:  # progress bar
        for episode in range(num_episodes):
            self.env.reset()

            episode_rewards = []
            episode_ER = []

            # Time is increased in steps, with the policy updating after every step.
            n_steps = int( np.ceil(len(self.env.time_data) / step_size) )
            
            for step_count in range(1, n_steps+1):
                
                # takes a step_size forward in time
                pred = self.env.step(step_size).squeeze(-1)  # state and action produced in step
                
                actual = self.temp_data[self.env.t_index:int(self.env.t_index + step_size), 0:self.env.n_rooms]

                # negative so reward can be maximised
                reward = -loss_fn(pred[:, 2:], actual)

                # Do gradient decent on sample
                ER, DR = self.update_policy(reward, self.env.RC.cooling_policy.log_probs)

                self.env.RC.cooling_policy.on_policy_reset()  # empty buffer

                # get last output and use for next initial value
                self.env.RC.iv = pred[-1, :].unsqueeze(1).detach()  # MUST DETACH GRAD

                episode_rewards.append(sum(reward))
                episode_ER.append(ER)

                self.env.t_index += int(step_size)  # increase environment time

                # update baseline as the mean of all discounted rewards in episode
                if self.baseline == 0:  # initial case is different
                    self.baseline = DR.mean() / step_count
                else:
                    self.baseline = (self.baseline*step_count + DR.mean()) / (step_count+1)
            
            # At end of episode set baseline to mean of final discounted rewards.
            # Carries over some info between episodes without
            self.baseline = DR.mean()

#             print(f'Episode {episode+1}, Expected Reward: {sum(episode_ER).item():.2f}, total_reward: {sum(episode_rewards).item():.2f}')

            total_ER.append(sum(episode_ER).detach())
            total_rewards.append(sum(episode_rewards).detach())

        return total_rewards, total_ER


# In[7]:


for i in range(1):
    policy = PolicyNetwork(7, 2)
    model = initialise_model(policy, scaling, weather_data_path)
    if load_model_path:
        model.load(load_model_path)# Reset loaded cooling policy with a blank one
        model.cooling_policy = PolicyNetwork(7, 2)
    env = LSIEnv(model, time_data)
    rl_b = Reinforce_baseline(env, time_data, temp_data)
    train_policy(rl_b, f'{opt_id}{i}')


# In[8]:


# for i in range(5):
#     policy = PolicyNetwork(7, 2)
#     model = initialise_model(policy, scaling, weather_data_path)
#     if load_model_path:
#         model.load(load_model_path)# Reset loaded cooling policy with a blank one
#         model.cooling_policy = PolicyNetwork(7, 2)
#     env = LSIEnv(model, time_data)
#     opt_id = 'NoBaseline'


#     rl = Reinforce(env, time_data, temp_data, alpha=1e-2)
#     train_policy(rl, f'{opt_id}{i}')


# In[9]:


import ray
import copy
import time

def get_rl_baseline():
    policy = PolicyNetwork(7, 2)
    model = initialise_model(policy, scaling, weather_data_path)
    if load_model_path:
        model.load(load_model_path)# Reset loaded cooling policy with a blank one
        model.cooling_policy = PolicyNetwork(7, 2)
#     env = LSIEnv(copy.deepcopy(model), time_data)
    
#     return Reinforce_baseline(copy.deepcopy(env), time_data, temp_data)
    env = LSIEnv(model, time_data)
    
    return Reinforce_baseline(env, time_data, temp_data)

def get_rl():
    policy = PolicyNetwork(7, 2)
    model = initialise_model(policy, scaling, weather_data_path)
    if load_model_path:
        model.load(load_model_path)# Reset loaded cooling policy with a blank one
        model.cooling_policy = PolicyNetwork(7, 2)
    env = LSIEnv(model, time_data)
    opt_id = 'NoBaseline'
    return Reinforce(env, time_data, temp_data, alpha=1e-2)



train_policy = ray.remote(train_policy)  # Make remote function


trials = []
opt_id = []
for i in range(5):
    trials.append(get_rl_baseline())
    opt_id.append(f'Baseline{i}')

for i in range(5):
    trials.append(get_rl())
    opt_id.append(f'NoBaseline{i}')


# train_policy(trials[0], f'{opt_id[0]}')
actors = [train_policy.remote(rl, f'{opt_id[i]}') for i, rl in enumerate(trials)]


results = ray.get([a for num, a in enumerate(actors)])

time.sleep(3)  # Print is cut short without sleep

ray.shutdown()


# In[13]:


trials[0].env.RC.cooling_policy.state_dict()['linear_relu_stack.0.weight']


# In[14]:


trials[1].env.RC.cooling_policy.state_dict()['linear_relu_stack.0.weight']


# In[10]:


ray.shutdown()


# In[ ]:





# In[ ]:




