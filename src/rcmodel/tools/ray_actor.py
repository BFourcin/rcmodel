import torch
import pandas as pd
import numpy as np
import ray
from matplotlib import pyplot as plt
from tqdm.auto import tqdm, trange
from pathlib import Path

from ..tools import BuildingTemperatureDataset, pltsolution_1rm,\
    exponential_smoothing, convergence_criteria
from ..physical import InputScaling


class RayActor:
    """
    Class provides basic framework to call ray for multiprocessing.
    Trains Physical then Policy one after the other.
    """
    def __init__(self, model, csv_path, physical_training=True, policy_training=True):
        self.model = model
        self.csv_path = csv_path
        self.physical_training = physical_training
        self.policy_training = policy_training

    def worker(self, opt_id, epochs=100):
        from ..optimisation.optimise_rc import OptimiseRC  # placing here avoids circular import

        def do_plots():
            if self.physical_training:
                # find smoothed curves:
                phys_smooth = exponential_smoothing(avg_train_loss_plot, conv_alpha, n=conv_win_len)


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

            if self.policy_training:
                pol_smooth = exponential_smoothing(rewards_plot, conv_alpha, n=conv_win_len)
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

        torch.set_num_threads(1)

        dir_path = f'./outputs/run{opt_id}/models/'  # where to save

        self.model.save(0, dir_path)  # save initial model

        dt = 30
        sample_size = 24 * 60 ** 2 / dt
        conv_win_len = 10  # MUST BE EVEN. total length of lookback window to measure convergence
        conv_alpha = 0.15

        op = OptimiseRC(self.model, self.csv_path, sample_size, dt, lr=1e-3, opt_id=opt_id)

        if self.policy_training:
            time_data = torch.tensor(pd.read_csv(self.csv_path, skiprows=0).iloc[:, 1], dtype=torch.float64)
            temp_data = torch.tensor(pd.read_csv(self.csv_path, skiprows=0).iloc[:, 2:].to_numpy(dtype=np.float32),
                                     dtype=torch.float32)

            env = LSIEnv(self.model, time_data)
            do_reinforce = Reinforce(env, time_data, temp_data, alpha=1e-2)

            rewards_plot = []
            ER_plot = []

        avg_train_loss_plot = []
        avg_test_loss_plot = []

        plot_results_data = BuildingTemperatureDataset(self.csv_path, 5 * sample_size, all=True)
        plot_dataloader = torch.utils.data.DataLoader(plot_results_data, batch_size=1, shuffle=False)

        # check if dir exists and make if needed
        Path(f'./outputs/run{opt_id}/plots/results/').mkdir(parents=True, exist_ok=True)

        for epoch in trange(epochs):

            if self.physical_training:
                # Physical Model Training:
                avg_train_loss = op.train()
                avg_train_loss_plot.append(avg_train_loss)

            if self.policy_training:
                # Policy Training:
                num_episodes = 1
                step_size = sample_size
                rewards, ER = do_reinforce.train(num_episodes, step_size)
                rewards_plot.append(rewards)
                ER_plot.append(ER)

            # Test Loss
            avg_test_loss = op.test()
            avg_test_loss_plot.append(avg_test_loss)

            # Save a plot of results every 10 epochs or if first/final epoch
            if (epoch + 1) % 10 == 0 or epoch == epochs - 1 or epoch == 0:
                pltsolution_1rm(self.model, plot_dataloader,
                                f'./outputs/run{opt_id}/plots/results/Result_Epoch{epoch + 1}.png')
                if epoch > 0:
                    do_plots()

            # Save Model
            model_id = epoch + 1
            self.model.save(model_id, dir_path)

            if self.policy_training:
                tqdm.write(f'Epoch {epoch + 1}, Train/Test Loss: {avg_train_loss:.2f}/{avg_test_loss:.2f}, Policy '
                           f'Rewards {sum(rewards).item():.1f}, Policy Expected Rewards {sum(ER).item():.1f}')
            else:
                tqdm.write(f'Epoch {epoch + 1}, Train/Test Loss: {avg_train_loss:.2f}/{avg_test_loss:.2f}')

        final_train_loss = avg_train_loss
        final_test_loss = avg_test_loss

        final_params = self.model.transform(self.model.params).detach().numpy()
        final_cooling = self.model.transform(self.model.loads).detach().numpy()

        cool_load = final_cooling[0, :]
        gain_load = final_cooling[1, :]

        return np.concatenate(([opt_id], final_params, gain_load, cool_load, [final_train_loss, final_test_loss]))


if __name__ == '__main__':
    from rcmodel import *

    use_ray = True

    epochs = 2

    if use_ray:
        num_cpus = 4
        num_jobs = num_cpus
        ray.init(num_cpus=num_cpus)

    # Initialise scaling class
    rm_CA = [100, 1e4]  # [min, max] Capacitance/area
    ex_C = [1e3, 1e8]  # Capacitance
    R = [0.1, 5]  # Resistance ((K.m^2)/W)
    Q_limit = [-100, 100]  # Cooling limit and gain limit in W/m2
    scaling = InputScaling(rm_CA, ex_C, R, Q_limit)

    # Laptop:
    weather_data_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather ' \
                        'Files/JuneSept.csv'
    csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/test2d_sorted.csv'

    # Hydra:
    # weather_data_path = '/home/benf/LSI/Data/Met Office Weather Files/JuneSept.csv'
    # csv_path = '/home/benf/LSI/Data/DummyData/train5d_sorted.csv'

    prior = PriorCoolingPolicy()
    model = initialise_model(prior, scaling, weather_data_path)

    if use_ray:
        RayActor = ray.remote(RayActor)
        actors = [RayActor.remote(model, csv_path, policy_training=False) for _ in range(num_jobs)]

        results = ray.get([a.worker.remote(num, epochs) for num, a in enumerate(actors)])

        ray.shutdown()

    else:
        actor = RayActor(model, csv_path)
        actor.policy_training = False  # Turn cooling optimisation off
        results = actor.worker(0, epochs)

    params_heading = ['Rm Cap/m2 (J/K.m2)', 'Ext Wl Cap 1 (J/K)', 'Ext Wl Cap 2 (J/K)', 'Ext Wl Res 1 (K.m2/W)',
                      'Ext Wl Res 2 (K.m2/W)', 'Ext Wl Res 3 (K.m2/W)', 'Int Wl Res (K.m2/W)', 'Offset Gain (W/m2)']
    cooling_heading = ['Cooling (W)']
    headings = [['Run Number'], params_heading, cooling_heading,
                ['Final Average Train Loss', 'Final Avg Test Loss']]
    flat_list = [item for sublist in headings for item in sublist]

    if use_ray:
        df = pd.DataFrame(np.array(results), columns=flat_list)
    else:
        df = pd.DataFrame([np.array(results)], columns=flat_list)

    df.to_csv('./outputs/results.csv', index=False, )

