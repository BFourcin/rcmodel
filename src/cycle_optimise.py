import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm, trange

from rcmodel import *

# Laptop
weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'

# Hydra:
# weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
# csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'  # where building data

# Load model?
load_model_path_physical = None  # or None
load_model_path_policy = None

# load_model_path_physical = './physical_model.pt'  # or None
# load_model_path_policy = './prior_policy.pt'

# opt_id = 0


def worker(opt_id):
    torch.set_num_threads(1)

    dir_path = f'./outputs/run{opt_id}/models/'  # where to save

    def do_plots():
        # find smoothed curves:
        phys_smooth = exponential_smoothing(avg_train_loss_plot, conv_alpha, n=conv_win_len)
        pol_smooth = exponential_smoothing(rewards_plot, conv_alpha, n=conv_win_len)

        # Plot Loss:
        fig, axs = plt.subplots(figsize=(10, 7))
        ax2 = axs.twinx()
        ln1 = axs.plot(range(1, len(avg_train_loss_plot) + 1), avg_train_loss_plot, label='train loss')
        ln2 = axs.plot(range(1, len(avg_test_loss_plot) + 1), avg_test_loss_plot, label='test loss')
        ln3 = axs.plot(range(1, len(phys_smooth) + 1), phys_smooth, 'g', label='train loss - smoothed')
        c = []
        for i in range(len(phys_smooth)):
            c.append(convergence_criteria(phys_smooth[0:i + 1], conv_win_len))

        ln4 = ax2.plot(range(1, len(phys_smooth) + 1), c, 'k--', label='convergence')

        fig.suptitle('Train and Test Loss During Training', fontsize=16)
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Mean Squared Error')
        axs.set_yscale('log')
        ax2.set_ylabel('Convergence')
        ax2.set_yscale('log')
        # legend:
        lns = ln1 + ln2 + ln3 + ln4
        labs = [l.get_label() for l in lns]
        axs.legend(lns, labs, loc=0)

        fig.savefig(f'./outputs/run{opt_id}/plots/LossPlot.png')
        plt.close()

        # Plot Rewards:
        fig, axs = plt.subplots(figsize=(10, 7))
        ax2 = axs.twinx()

        y = np.array(rewards_plot, dtype=np.float32)
        ln1 = axs.plot(range(1, len(y) + 1), -y, label='total rewards')
        ln2 = axs.plot(range(1, len(pol_smooth) + 1), -np.array(pol_smooth, dtype=np.float32), 'g',
                       label='rewards - smoothed')
        c = []
        for i in range(len(pol_smooth)):
            c.append(convergence_criteria(pol_smooth[0:i + 1], conv_win_len))

        ln3 = ax2.plot(range(1, len(pol_smooth) + 1), c, 'k--', label='convergence')

        # legend:
        lns = ln1 + ln2 + ln3
        labs = [l.get_label() for l in lns]
        axs.legend(lns, labs, loc=0)

        fig.suptitle('Rewards', fontsize=16)
        axs.set_xlabel('Epoch')
        axs.set_ylabel('Reward')
        axs.set_yscale('log')

        ax2.set_ylabel('Convergence')
        ax2.set_yscale('log')

        fig.savefig(f'./outputs/run{opt_id}/plots/RewardsPlot.png')
        plt.close()

    def avg_last_values(y, n):
        """
        Get the mean of the last n valid values from a mixed list of numbers and None.

        If y[-n] contains None nothing is returned.
        Otherwise, y[-n].mean() is returned.
        """

        for i, val in enumerate(reversed(y)):  # look along reversed list and stop at the first None
            if val is None:
                break
        if i > n:  # if there are enough non-None values:
            y_hat = [np.array(y[-n]).mean()]
            return y_hat

    def init_scaling():
        # Initialise scaling class
        rm_CA = [100, 1e4]  # [min, max] Capacitance/area
        ex_C = [1e3, 1e8]  # Capacitance
        R = [0.1, 5]  # Resistance ((K.m^2)/W)
        Q_limit = [300]  # Cooling limit and gain limit in W/m2
        scaling = InputScaling(rm_CA, ex_C, R, Q_limit)
        return scaling

    # Initialise Model
    scaling = init_scaling()
    policy = PolicyNetwork(5, 2)
    model = initialise_model(policy, scaling, weather_data_path)

    # load physical and/or policy models if available
    if load_model_path_policy:
        model.load(load_model_path_policy)  # load policy
        model.init_physical()  # re-randomise physical params, as they were also copied from the loaded policy

    if load_model_path_physical:
        # m = initialise_model(None, scaling, weather_data_path)  # dummy model to load physical params on to.
        m = initialise_model(PolicyNetwork(5, 2), scaling, weather_data_path)  # dummy model to load physical params on to.
        m.load(load_model_path_physical)
        model.params = m.params  # put loaded physical parameters onto model.
        model.loads = m.loads
        del m

    model.save(0, dir_path)  # save initial model

    # # Initialise Optimise Class - for training physical model
    dt = 30
    sample_size = 24 * 60 ** 2 / dt  # ONE DAY
    op = OptimiseRC(model, csv_path, sample_size, dt, lr=1e-2, opt_id=opt_id)

    # Initialise Reinforcement Class - for training policy
    time_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 1], dtype=torch.float64)
    temp_data = torch.tensor(pd.read_csv(csv_path, skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32), dtype=torch.float32)
    env = LSIEnv(model, time_data, temp_data)
    rl = Reinforce(env, alpha=1e-2)

    # lists to keep track of process, used for plot at the end
    avg_train_loss_plot = []
    avg_test_loss_plot = []
    rewards_plot = []
    ER_plot = []
    count = 0  # Counts number of epochs since start.

    # initialise variables to keep Pycharm happy:
    epoch = None
    rewards = None

    # Dataloader to be used when plotting a model run. This is just for info.
    plot_results_data = BuildingTemperatureDataset(csv_path, 5 * sample_size, all=True)
    plot_dataloader = torch.utils.data.DataLoader(plot_results_data, batch_size=1, shuffle=False)

    # check if dir exists and make if needed
    Path(f'./outputs/run{opt_id}/plots/results/').mkdir(parents=True, exist_ok=True)
    Path(f'./outputs/run{opt_id}/plots/policy_img/').mkdir(parents=True, exist_ok=True)

    start_num = 0  # Number of cycles to start at. Used if resuming the run. i.e. the first cycle is (start_num + 1)

    # Convergence happens when the convergence criteria is < tol.
    # See convergence_criteria() for more info.
    tol_phys = 0.005  # 0.5%
    tol_policy = 0.005  # 0.05%
    conv_win_len = 10  # MUST BE EVEN. total length of lookback window to measure convergence
    conv_alpha = 0.15

    cycles = 2
    max_epochs = 3

    plt.ioff()  # Reduces memory usage by matplotlib
    for cycle in trange(cycles):

        # -------- Physical model training --------
        tqdm.write('Physical Model Training:')
        y_hat = None  # reset value
        convergence = torch.inf
        for epoch in range(max_epochs):
            avg_train_loss = op.train()
            avg_train_loss_plot.append(avg_train_loss)

            # Test Loss
            avg_test_loss = op.test()
            avg_test_loss_plot.append(avg_test_loss)

            # Add dummy value to keep continuity in the graphs.
            rewards_plot.append(None)
            ER_plot.append(None)

            # Save Model
            model_id = count + start_num
            model.save(model_id, dir_path)
            count += 1

            # initialise y_hat if needed after calc the convergence by comparing a window of smoothed results.
            if y_hat is None:
                y_hat = avg_last_values(avg_train_loss_plot, conv_win_len)
            else:
                y_hat = exponential_smoothing(avg_train_loss, conv_alpha, y_hat, n=conv_win_len)
                convergence = convergence_criteria(y_hat, conv_win_len)
                if convergence is None:
                    convergence = torch.inf  # Just to allow the print statement below

            tqdm.write(
                f'Epoch {count}, Train/Test Loss: {avg_train_loss:.2f}/{avg_test_loss:.2f}, Convergence/Cutoff: {convergence:.3f}/{tol_phys:.3f}')

            # check if percentage change from mean is less than tol. i.e. convergence
            if convergence < tol_phys:
                tqdm.write(f'Physical converged in {epoch + 1} epochs. Total epochs: {count}\n')
                break

        if epoch+1 == max_epochs and convergence > tol_phys:
            tqdm.write(f'Failed to converge, max epochs reached. Total epochs: {count}\n')

        # Save a plot of results after physical training
        with torch.no_grad():
            pltsolution_1rm(model, plot_dataloader,
                            f'./outputs/run{opt_id}/plots/results/Result_Cycle{start_num + cycle + 1}a.png')

        # -------- Policy training --------
        tqdm.write('Policy Training:')
        y_hat = None  # reset value
        convergence = torch.inf
        for epoch in range(max_epochs):
            rewards, ER = rl.train(1, sample_size)

            rewards = torch.tensor(rewards).sum().item()
            ER = torch.tensor(ER).sum().item()

            rewards_plot.append(rewards)
            ER_plot.append(ER)

            # Add dummy value to keep continuity in the graphs.
            avg_train_loss_plot.append(None)
            avg_test_loss_plot.append(None)

            # Save Model
            model_id = count + start_num
            model.save(model_id, dir_path)
            count += 1

            # initialise y_hat if needed after calc the convergence by comparing a window of smoothed results.
            if y_hat is None:
                y_hat = avg_last_values(rewards_plot, conv_win_len)
            else:
                y_hat = exponential_smoothing(rewards, conv_alpha, y_hat, n=conv_win_len)
                convergence = convergence_criteria(y_hat, conv_win_len)
                if convergence is None:
                    convergence = torch.inf  # Just to allow the print statement below

            tqdm.write(
                f'Epoch {count}, Rewards/Expected Rewards: {rewards:.2f}/{ER:.2f}, Convergence/Cutoff: {convergence:.3f}/{tol_policy:.3f}')

            # check if percentage change from mean is less than tol. i.e. convergence
            if convergence < tol_policy:
                tqdm.write(f'Policy converged in {epoch + 1} epochs. Total epochs: {count}\n')
                break

        if epoch+1 == max_epochs and convergence > tol_policy:
            tqdm.write(f'Failed to converge, max epochs reached. Total epochs: {count}\n')

        with torch.no_grad():  # makes plotting 20% faster
            # Save a plot of results after policy training
            pltsolution_1rm(model, plot_dataloader, f'./outputs/run{opt_id}/plots/results/Result_Cycle{start_num + cycle + 1}b.png')
            # Plot loss and reward plot
            do_plots()
            policy_image(model.cooling_policy, path=f'./outputs/run{opt_id}/plots/policy_img/policy{start_num + cycle + 1}.png')

        # save outputs to .csv:
        pd.DataFrame(rewards_plot).to_csv(f'./outputs/run{opt_id}/plots/rewards.csv', index=False)
        pd.DataFrame(avg_train_loss_plot).to_csv(f'./outputs/run{opt_id}/plots/train_loss.csv', index=False)
        pd.DataFrame(avg_test_loss_plot).to_csv(f'./outputs/run{opt_id}/plots/test_loss.csv', index=False)

    final_params = model.transform(model.params).detach().numpy()
    final_cooling = model.transform(model.loads[0, :]).detach().numpy()
    final_gain = model.transform(model.loads[1, :]).detach().numpy()

    return np.concatenate(([opt_id], final_params, final_gain, final_cooling, [rewards]))


if __name__ == '__main__':
    import ray

    worker(0)

    num_cpus = 3
    num_jobs = num_cpus
    ray.init(num_cpus=num_cpus)

    worker = ray.remote(worker)

    results = ray.get([worker.remote(num) for num in range(num_jobs)])
    ray.shutdown()

    params_heading = ['Rm Cap/m2 (J/K.m2)', 'Ext Wl Cap 1 (J/K)', 'Ext Wl Cap 2 (J/K)', 'Ext Wl Res 1 (K.m2/W)',
                      'Ext Wl Res 2 (K.m2/W)', 'Ext Wl Res 3 (K.m2/W)', 'Int Wl Res (K.m2/W)', 'Offset Gain (W/m2)']

    cooling_heading = ['Cooling (W)']
    headings = [['Run Number'], params_heading, cooling_heading,
                ['Final Reward']]
    flat_list = [item for sublist in headings for item in sublist]

    df = pd.DataFrame(np.array(results), columns=flat_list)

    df.to_csv('./outputs/results.csv', index=False, )

