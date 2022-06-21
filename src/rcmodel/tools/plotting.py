import torch
import pandas as pd
import matplotlib.pyplot as plt


# helper functions:
def pltsolution_1rm(model, dataloader=None, filename=None, prediction=None, time=None):
    """
    Runs model and plots the first sample from a given dataloader.
    pltsolution_1rm(model, dataloader, filename)

    Or

    Plot a pre run prediction. dt is used to produce the x-axis
    pltsolution_1rm(model, filename, prediction, time)
    """
    model.reset_iv()  # Reset initial value
    model.eval()  # Put model in evaluation mode

    # Get solution ---------------
    if dataloader:
        time, data_temp = next(iter(dataloader))

        time = time.squeeze(0)
        data_temp = data_temp.squeeze(0)

        t_days = (time - time.min()) / (24 * 60 ** 2)  # Time in days

        prediction = model(time)
        prediction = prediction.squeeze(-1)

    else:
        t_days = (time - time.min()) / (24 * 60 ** 2)  # Time in days
        data_temp = (t_days * torch.nan).unsqueeze(0).T

    # Get Heating Control for each room
    if model.cooling_policy:
        record_action = torch.tensor(model.record_action)
        Q_tdays = record_action[:, 0] / (24 * 60 ** 2)  # Time in days
        Q_on_off = record_action[:, 1:]  # Cooling actions

        Q_area = model.transform(model.loads)
        Q_area = model.scaling.physical_loads_scaling(Q_area)
        Q_watts = model.building.proportional_heating(Q_area)  # convert from W/m2 to W

        Q = Q_on_off * -Q_watts.unsqueeze(1)  # Q is timeseries of Watts for each room.

    else:
        Q_tdays = t_days
        Q = torch.zeros(len(Q_tdays))

    gain = model.scaling.physical_loads_scaling(model.transform(model.loads))[1, :]
    gain_watts = gain * model.building.rooms[0].area

    # Compute and print loss.
    # loss_fn = torch.nn.MSELoss()
    # num_cols = len(model.building.rooms)  # number of columns to take from data.
    # loss = loss_fn(prediction[:, 2:], data_temp[:, 0:num_cols])

    # print(f"Test loss = {loss.item():>8f}")

    # ---------------------------------

    # Plot Solution

    ax2ylim = 250

    fig, axs = plt.subplots(figsize=(10, 8))

    ax2 = axs.twinx()
    ln1 = axs.plot(t_days.detach().numpy(), prediction[:, 2:].detach().numpy(), label=r'model ($^\circ$C)')
    ln2 = axs.plot(t_days.detach().numpy(), data_temp[:, 0].detach().numpy(), label=r'data ($^\circ$C)')
    ln3 = axs.plot(t_days.detach().numpy(), model.Tout_continuous(time).detach().numpy(), label=r'outside ($^\circ$C)')
    ln4 = ax2.plot(Q_tdays.detach().numpy(), Q.detach().numpy(), '--', color='black', alpha=0.5, label='heat ($W$)')
    ln5 = ax2.axhline(gain_watts.detach().numpy(), linestyle='-.', color='black', alpha=0.5, label='gain ($W$)')
    axs.set_title(model.building.rooms[0].name)
    ax2.set_ylabel(r"Heating/Cooling ($W$)")
    # ax2.set_ylim(-ax2ylim, ax2ylim)

    lns = ln1 + ln2 + ln3 + ln4 + [ln5]  # for some reason ln5 isn't auto put into a list
    labs = [l.get_label() for l in lns]
    axs.legend(lns, labs, loc=0)

    axs.set(xlabel='Time (days)', ylabel=r'Temperature ($^\circ$C)')

    if filename:
        fig.savefig(filename)
        plt.close()

    else:
        plt.show()
