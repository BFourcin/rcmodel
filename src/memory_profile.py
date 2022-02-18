# info check this out:
# https://pypi.org/project/memory-profiler/
import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm, trange
import ray
from rcmodel import *

from memory_profiler import profile


@profile
def mem_profile():


    # Laptop
    weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
    csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'

    # Hydra:
    # weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
    # csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'  # where building data

    # Load model?
    load_model_path_physical = './physical_model.pt'  # or None
    load_model_path_policy = './prior_policy.pt'

    # opt_id = 0


    def worker(opt_id):
        torch.set_num_threads(1)

        dir_path = f'./outputs/run{opt_id}/models/'  # where to save

        def do_plots():
            fig = plt.figure(figsize=(10, 7), dpi=400)
            plt.plot(range(1, len(avg_train_loss_plot) + 1), avg_train_loss_plot, label='train loss')
            plt.plot(range(1, len(avg_test_loss_plot) + 1), avg_test_loss_plot, label='test loss')
            plt.title('Train and Test Loss During Training')
            plt.xlabel('Epoch')
            plt.ylabel('Mean Squared Error')
            plt.legend()
            fig.savefig(f'./outputs/run{opt_id}/plots/LossPlot.png')
            plt.close()

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
            Q_limit = [-10000, 10000]
            scaling = InputScaling(rm_CA, ex_C, R, Q_limit)
            return scaling

        # Initialise Model
        scaling = init_scaling()
        policy = PolicyNetwork(5, 2)
        model = initialise_model(policy, scaling, weather_data_path)

        model.save(0, dir_path)  # save initial model

        # Initialise Optimise Class - for training physical model
        dt = 30
        sample_size = 24 * 60 ** 2 / dt  # ONE DAY
        op = OptimiseRC(model, csv_path, sample_size, dt, lr=1e-3, opt_id=opt_id)

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
        count = 1  # Counts number of epochs since start.

        # Dataloader to be used when plotting a model run. This is just for info.
        plot_results_data = BuildingTemperatureDataset(csv_path, 5 * sample_size, all=True)
        plot_dataloader = torch.utils.data.DataLoader(plot_results_data, batch_size=1, shuffle=False)

        # check if dir exists and make if needed
        Path(f'./outputs/run{opt_id}/plots/results/').mkdir(parents=True, exist_ok=True)

        # load physical and/or policy models if available
        if load_model_path_policy:
            model.load(load_model_path_policy)  # load policy
            model.init_physical()  # re-randomise physical params, as they were also copied from the loaded policy

        if load_model_path_physical:
            m = initialise_model(None, scaling, weather_data_path)  # dummy model to load physical params on to.
            m.load(load_model_path_physical)
            model.params = m.params  # put loaded physical parameters onto model.
            del m

        start_num = 0  # Number of cycles to start at. Used if resuming the run. i.e. the first cycle is (start_num + 1)

        # Convergence happens when the mean gradient of loss/reward is < tol.
        tol_phys = 0.01  # 1%
        tol_policy = 0.02  # 2%
        window_len = 10.  # Length of convergence look-back window.

        cycles = 5
        max_epochs = 50

        plt.ioff()  # Reduces memory usage by matplotlib
        for cycle in trange(cycles):

            # -------- Physical model training --------
            tqdm.write('Physical Model Training:')
            r_prev = (torch.arange(window_len) + 1) * 10  # dummy rewards to prevent convergence on first epochs
            loss_prev = 0
            for epoch in range(max_epochs):
                avg_train_loss = op.train()
                avg_train_loss_plot.append(avg_train_loss)

                # Get difference from previous
                indx = epoch % len(r_prev)  # cycles the indexes in array
                r_prev[indx] = avg_train_loss  # keeps track of a window of differences

                # Test Loss
                avg_test_loss = op.test()
                avg_test_loss_plot.append(avg_test_loss)

                tqdm.write(
                    f'Epoch {count}, Train/Test Loss: {avg_train_loss:.2f}/{avg_test_loss:.2f}, Mean diff: {abs((r_prev.mean() - r_prev[indx]) / r_prev[indx]) * 100:.3f} %')

                # Save Model
                model_id = count + start_num
                model.save(model_id, dir_path)
                count += 1

                # check if percentage change from mean is less than tol. i.e. convergence
                if abs((r_prev.mean() - r_prev[indx]) / r_prev[indx]) < tol_phys:
                    break

            tqdm.write(f'Physical converged in {epoch + 1} epochs. Total epochs: {count - 1}\n')

            # Save a plot of results after physical training
            pltsolution_1rm(model, plot_dataloader,
                            f'./outputs/run{opt_id}/plots/results/Result_Cycle{start_num + cycle + 1}a.png')

            # -------- Policy training --------
            tqdm.write('Policy Training:')
            r_prev = (torch.arange(window_len) + 1) * 1000
            for epoch in range(max_epochs):
                rewards, ER = rl.train(1, sample_size)

                indx = epoch % len(r_prev)
                r_prev[indx] = torch.tensor(rewards)  # keeps track of a window of differences
                rewards_plot.append(rewards)
                ER_plot.append(ER)

                tqdm.write(
                    f'Epoch {count}, Rewards/Expected Rewards: {torch.tensor(rewards).sum().item():.2f}/{torch.tensor(ER).sum().item():.2f}, Mean diff: {abs((r_prev.mean() - r_prev[indx]) / r_prev[indx]) * 100:.3f} %')

                # Save Model
                model_id = count + start_num
                model.save(model_id, dir_path)
                count += 1

                # check if percentage change from mean is less than tol. i.e. convergence
                if abs((torch.mean(r_prev) - r_prev[indx]) / r_prev[indx]) < tol_policy:
                    break

            tqdm.write(f'Policy converged in {epoch + 1} epochs. Total epochs: {count - 1}\n')

            # Save a plot of results after policy training
            pltsolution_1rm(model, plot_dataloader, f'./outputs/run{opt_id}/plots/results/Result_Cycle{start_num + cycle + 1}b.png')

            # Plot loss and reward plot
            do_plots()

        final_params = model.transform(model.params).detach().numpy()
        final_cooling = model.transform(model.cooling).detach().numpy()

        return np.concatenate(([opt_id], final_params, final_cooling, [torch.tensor(rewards).sum().item()]))



    num_cpus = 2
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


if __name__ == '__main__':

    mem_profile()