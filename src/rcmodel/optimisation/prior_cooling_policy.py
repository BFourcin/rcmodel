import torch
import time


class PriorCoolingPolicy:
    """
    Class to provide a baseline cooling policy.

    Parameters
    ----------
    temp_on : float,
        Room temperature which when surpassed causes cooling to turn on.

    temp_off : float,
        Room temperature which when dropped below causes cooling to turn off.

    time_on : float,
        Beginning of cooling window. Time in the format: 9.25 is (09:00), 23.5 is (23:30).

    time_off : float,
        End of cooling window. Time in the format: 9.25 is (09:00), 23.5 is (23:30).
    """

    def __init__(self, temp_on=23, temp_off=21, time_on=9, time_off=17):
        self.temp_on = temp_on
        self.temp_off = temp_off
        self.time_on = time_on
        self.time_off = time_off

        self.cooling_on = False
        self.training = False  # set training mode off so model doesn't start trying to store log_probs.

    def get_action(self, x, unix_time):
        """
        Policy performs a number of logic checks and outputs an action, cooling on or off (1 or 0).

        Standard thermometer logic is used, cooling can turn on if time is within the schedule.
        Default between 09:00 - 17:00, Mon-Fri.
        If time is within the schedule cooling is triggered when room temp is higher than the set point (default 23),
        it will then only turn off when temperature drops below another set point (default 21).

        Parameters
        ----------
        x : torch.tensor,
            Temperature output of the rooms in the model. Do not include the latent wall temperatures,
            len(x) == len(building.rooms)

        unix_time : float or int,
            Unix epoch time.

        Returns
        -------
        int: 1 or 0, cooling on or off
        """

        if x.ndim == 0:
            x = x.unsqueeze(0)

        action = torch.zeros(len(x))

        for i, rm_temp in enumerate(x):

            if self.time_on > 24 or self.time_off > 24:
                raise ValueError('Ensure time_on and time_off are valid.')

            if self.time_off < self.time_on:
                raise ValueError('time_off should be later than time_on')

            # First see if time is within the valid period when cooling can be on:
            day, num_time = get_time(unix_time)

            # if not weekend
            if day != 'Saturday' and day != 'Sunday':

                # if time is between set points
                if self.time_on < num_time < self.time_off:

                    # if room is hot turn cooling on
                    if rm_temp > self.temp_on:
                        self.cooling_on = True
                        action[i] = 1
                        continue

                    # if cooling has already been turned on keep it on until temp is below temp_off
                    elif rm_temp > self.temp_off and self.cooling_on:
                        action[i] = 1
                        continue

            # Only reached if criteria above not met.
            self.cooling_on = False
            # action already zero so can remain unchanged

        log_prob = torch.log(torch.tensor(1))  # Dummy value, enables imitation of actual policy

        return action, log_prob

    def eval(self):
        """Dummy function to enable policy.eval()"""
        pass


def get_time(t):
    """
    Convert Unix Epoch time to weekday and clock time.

    Parameters
    ----------
    t: float or int,
        Unix epoch time.

    Returns
    -------
    strings giving the converted day of week, hour and minute
    """
    if torch.is_tensor(t):
        t = t.item()

    day = time.strftime('%A', time.localtime(t))
    hr = time.strftime('%H', time.localtime(t))
    minute = time.strftime('%M', time.localtime(t))

    num_time = int(hr) + int(minute)/60

    return day, num_time


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Plots two weeks of actions w
    start = 1624834800
    week = 14 * 24 * 60**2

    x = torch.tensor([0, 0, 23, 22, 22])

    tn = torch.arange(start, start + week, 60)

    actions = torch.zeros((len(tn), len(x)))

    prior = PriorCoolingPolicy()

    for i, t in enumerate(tn):
        actions[i, :], _ = prior.get_action(x, t)

    plt.plot(tn, actions)
    plt.show()

    print(actions)
