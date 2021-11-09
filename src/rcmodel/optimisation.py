from filelock import FileLock
import pandas as pd


def train(model, device, dataloader, optimizer):
    """
    Performs one epoch of training.
    Order of rooms in building and in data must match otherwise model will fit wrong rooms to data.
    """
    model.train()
    num_cols = len(model.building.rooms)  # number of columns to use from data.
    num_batches = len(dataloader)
    train_loss = 0
    loss_fn = torch.nn.MSELoss()

    for batch, (time, temp) in enumerate(dataloader):
        time, temp = time.to(device), temp.to(device)  # Put on GPU if available

        # Get model arguments:
        time = time.squeeze(0)
        temp = temp.squeeze(0)

        # Compute prediction and loss
        pred = model(time)
        pred = pred.squeeze(-1)  # change from column to row matrix

        loss = loss_fn(pred[:, 2:], temp[:, 0:num_cols])
        train_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return train_loss / num_batches


def test(model, device, dataloader):
    model.eval()
    num_batches = len(dataloader)
    num_cols = len(model.building.rooms)  # number of columns to take from data.
    test_loss = 0
    loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        for (time, temp) in dataloader:
            time, temp = time.to(device), temp.to(device)  # Put on GPU if available

            time = time.squeeze(0)
            temp = temp.squeeze(0)

            pred = model(time)
            pred = pred.squeeze(-1)  # change from column to row matrix
            test_loss += loss_fn(pred[:, 2:], temp[:, 0:num_cols]).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss


def dataset_creator(path, sample_size, dt):
    path_sorted = sort_data(path, dt)
    with FileLock("./data.lock"):
        training_data = BuildingTemperatureDataset(path_sorted, sample_size, train=True)
        train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=1, shuffle=False)
        test_data = BuildingTemperatureDataset(path_sorted, sample_size, test=True)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
        
    return train_dataloader, test_dataloader


def sort_data(path, dt):
    """
    Check if path has sorted data tag (_sorted)
    If not check if data has previously been sorted and exists in the directory.
    Check to see if the value dt is correct
    If not sort data and write filename_sorted.csv

    data is sorted by time in ascending order and downsampled to a frequency of dt seconds.
    Missing values are interpolated.
    A time-date string is also inserted.
    """
    def sort(path, dt):
        df = pd.read_csv(path)

        if path[-11:] == '_sorted.csv':
            path_sorted = path
        else:
            path_sorted = path[:-4] + '_sorted.csv'

        # Sort df by time (raw data not always in order)
        df = df.sort_values(by=["time"], ascending=True)

        # insert date-time value at start of df
        try:
            df.insert(loc=0, column='date-time', value=pd.to_datetime(df['time'], unit='ms'))
        except ValueError:
            raise ValueError('Data appears to have already been sorted. Check if still appropriate and add _sorted.csv tag to avoid this error.')

        # downscale data to a frequency of dt (seconds) use the mean value and round to 2dp.
        df = df.set_index('date-time').resample(str(dt) + 's').mean().round(2)

        # time column is converted to unix epoch seconds to match the date-time
        df["time"] = (df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")

        # change date-time from UTC to Local time
        df = df.tz_localize('Europe/London')

        df = df.interpolate().round(2)  # interpolate missing values NaN

        df.to_csv(path_sorted, index=True)

    def need_to_sort(path, dt):

        def get_dt(path):
            df_dt = pd.read_csv(path)['time'][0:2].values
            return df_dt[1] - df_dt[0]

        # Does path already have sorted tag?
        if path[-11:] == '_sorted.csv':

            # if so, is dt correct?
            if get_dt(path) == dt:

                return False  # path and file is correct dont sort

            else:
                return True  # dt is wrong, re-sort

        # path does not contain _sorted.csv
        else:

            # Does path_sorted exist?
            path_sorted = path[:-4] + '_sorted.csv'
            import os.path
            if os.path.isfile(path_sorted):  # check if file already exists

                # if file exists check if dt is correct
                if get_dt(path_sorted) == dt:
                    return False  # correct file already exists don't sort
                else:
                    return True  # file exists but dt wrong, re-sort

            else: # File doesn't exist
                return True

    if need_to_sort(path, dt):
        sort(path, dt)

    # return the path_sorted
    if path[-11:] == '_sorted.csv':
        path_sorted = path
    else:
        path_sorted = path[:-4] + '_sorted.csv'

    return path_sorted


class OptimiseRC:
    """
    Parameters
    ----------
    model : object
        RCModel class object.
    csv_path : string
        Path to .csv file containing room temperature data.
        Data will be sorted if not done already and saved to a new file with the tag '_sorted'
    sample_size : int
        Length of indexes to sample from dataset per batch.
    dt : int
        Timestep data will be resampled to.
    lr : float
        Learning rate for optimiser.

    see https://docs.ray.io/en/latest/using-ray-with-pytorch.html
    """
    def __init__(self, model, csv_path, sample_size, dt=30, lr=1e-3):
        self.model = model
        self.model.init_params()  # randomise parameters
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_dataloader, self.test_dataloader = dataset_creator(csv_path, int(sample_size), int(dt))
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self):
        train(self.model, self.device, self.train_dataloader, self.optimizer)
        return test(self.model, self.device, self.test_dataloader)

    def train_loop(self, epochs):
        print(self.model.params)
        for i in range(int(epochs)):
            print(f"Epoch {i + 1}\n-------------------------------")
            testloss = self.train()

        results = [testloss, self.model]
        return results

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def save(self):
        torch.save(self.model.state_dict(), "rcmodel_params.pt")


def example_op():
    import torch
    from rcmodel.room import Room
    from rcmodel.building import Building
    # from rcmodel.RCModel import RCModel
    from RCModel import RCModel
    from rcmodel.tools import InputScaling
    from rcmodel.tools import BuildingTemperatureDataset
    from xitorch.interpolate import Interp1D
    import time


    def initialise_model(pi):
        torch.cuda.is_available = lambda: False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def change_origin(coords):
            x0 = 92.07
            y0 = 125.94

            for i in range(len(coords)):
                coords[i][0] = round((coords[i][0] - x0) / 10, 2)
                coords[i][1] = round((coords[i][1] - y0) / 10, 2)

            return coords

        capacitance = 3000  # Variable changed later
        rooms = []

        name = "seminar_rm_a_t0106"
        coords = change_origin(
            [[92.07, 125.94], [92.07, 231.74], [129.00, 231.74], [154.45, 231.74], [172.64, 231.74], [172.64, 125.94]])
        rooms.append(Room(name, capacitance, coords))

        # Initialise Building
        height = 1
        Re = [4, 1, 0.55]  # Sum of R makes Uval=0.18 #Variable changed later
        Ce = [1.2 * 10 ** 3, 0.8 * 10 ** 3]  # Variable changed later
        Rint = 0.66  # Uval = 1/R = 1.5 #Variable changed later

        bld = Building(rooms, height, Re, Ce, Rint)

        rm_CA = [200, 800]  # [min, max] Capacitance/area
        ex_C = [1.5 * 10 ** 4, 10 ** 6]  # Capacitance
        R = [0.2, 1.2]  # Resistance ((K.m^2)/W)

        scaling = InputScaling(rm_CA, ex_C, R)
        scale_fn = scaling.physical_scaling  # function to scale parameters back to physical values

        path_Tout = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
        df = pd.read_csv(path_Tout)
        Tout = torch.tensor(df['Hourly Temperature (°C)'], device=device)
        t = torch.tensor(df['time'], device=device)

        Tout_continuous = Interp1D(t, Tout, method='linear')

        # Initialise RCOptimModel with the building
        # transform = torch.sigmoid
        transform = torch.sigmoid
        model = RCModel(bld, scaling, Tout_continuous, transform, pi)
        model.to(device)  # put model on GPU if available
        model.Q_lim = 10000

        return model, Tout_continuous

    model, _ = initialise_model(None)
    csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'
    dt = 30
    sample_size = 2*60**2/dt
    start = time.time()
    op = OptimiseRC(model, csv_path, sample_size, dt)

    epochs = 5

    for i in range(epochs):
        print(f"Epoch {i + 1}\n-------------------------------")
        op.train()

    print(f'duration {time.time()-start:.1f} s')


# @ray.remote
# class RayActor:
#     def __init__(self, model, epochs=50, lr=1e-3):
#         self.model = model
#
#     # ----- Training Loop -----
#     def worker(self):
#         torch.set_num_threads(1)
#         model.init_params()
#
#         model.Q_lim = 10000
#
#         start = time.time()
#
#         # hyperparameters:
#         epochs = 60
#         learning_rate = 4e-1
#         loss_fn = torch.nn.MSELoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#         # training loop:
#         for i in range(self.epochs):
#             trainloss = train_loop(train_dataloader, model, loss_fn, optimizer)
#
#             if (i + 1) % 5 == 0 or i == 0:
#                 print('runtime: ', round(time.time() - start, 2), '| Epoch ' + str(i + 1) + '/' + str(epochs),
#                       '  Avg Train Loss: ', round(trainloss, 2))
#
#             if i == self.epochs - 1:
#                 testloss = test_loop(test_dataloader, model, loss_fn)
#
#         print(testloss)
#
#         params, heating = model.params, model.heating
#
#         results = [params, heating, testloss, model]
#
#         return results
#
#
# if __name__ == '__main__':
#
#     num_cpus = 85
#     num_jobs = num_cpus
#
#     ray.init(num_cpus=num_cpus)
#
#     start = time.time()
#
#     actors = [Ray_Actor.remote() for _ in range(num_jobs)]
#
#     results = ray.get([a.worker.remote() for a in actors])
#
#     #     results = ray.get([worker.remote(x) for x in range(num_jobs)])
#
#     print("duration =", time.time() - start)
#     ray.shutdown()
#
#     # check if dir exists and make if needed
#     from pathlib import Path
#
#     Path("./models/").mkdir(parents=True, exist_ok=True)
#
#     # Save model from every run
#     for i in range(len(results)):
#         torch.save(results[i][3].state_dict(), './models/torchmodel' + str(i + 1) + '.pth')


if __name__ == '__main__':

    import ray
    import torch
    from rcmodel.room import Room
    from rcmodel.building import Building
    # from rcmodel.RCModel import RCModel
    from RCModel import RCModel
    from rcmodel.tools import InputScaling
    from rcmodel.tools import BuildingTemperatureDataset
    from xitorch.interpolate import Interp1D
    import time


    def initialise_model(pi):
        torch.cuda.is_available = lambda: False
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        def change_origin(coords):
            x0 = 92.07
            y0 = 125.94

            for i in range(len(coords)):
                coords[i][0] = round((coords[i][0] - x0) / 10, 2)
                coords[i][1] = round((coords[i][1] - y0) / 10, 2)

            return coords

        capacitance = 3000  # Variable changed later
        rooms = []

        name = "seminar_rm_a_t0106"
        coords = change_origin(
            [[92.07, 125.94], [92.07, 231.74], [129.00, 231.74], [154.45, 231.74], [172.64, 231.74], [172.64, 125.94]])
        rooms.append(Room(name, capacitance, coords))

        # Initialise Building
        height = 1
        Re = [4, 1, 0.55]  # Sum of R makes Uval=0.18 #Variable changed later
        Ce = [1.2 * 10 ** 3, 0.8 * 10 ** 3]  # Variable changed later
        Rint = 0.66  # Uval = 1/R = 1.5 #Variable changed later

        bld = Building(rooms, height, Re, Ce, Rint)

        rm_CA = [200, 800]  # [min, max] Capacitance/area
        ex_C = [1.5 * 10 ** 4, 10 ** 6]  # Capacitance
        R = [0.2, 1.2]  # Resistance ((K.m^2)/W)

        scaling = InputScaling(rm_CA, ex_C, R)
        scale_fn = scaling.physical_scaling  # function to scale parameters back to physical values

        path_Tout = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/Met Office Weather Files/JuneSept.csv'
        df = pd.read_csv(path_Tout)
        Tout = torch.tensor(df['Hourly Temperature (°C)'], device=device)
        t = torch.tensor(df['time'], device=device)

        Tout_continuous = Interp1D(t, Tout, method='linear')

        # Initialise RCOptimModel with the building
        # transform = torch.sigmoid
        transform = torch.sigmoid
        model = RCModel(bld, scaling, Tout_continuous, transform, pi)
        model.to(device)  # put model on GPU if available
        model.Q_lim = 10000

        return model, Tout_continuous


    model, _ = initialise_model(None)
    csv_path = '/Users/benfourcin/OneDrive - University of Exeter/PhD/LSI/Data/DummyData/train5d_sorted.csv'
    dt = 30
    sample_size = 2 * 60 ** 2 / dt

    num_cpus = 5
    num_jobs = num_cpus

    ray.init(num_cpus=num_cpus)

    start = time.time()

    RemoteOptimise = ray.remote(OptimiseRC)  # need if don't use @ray.remote

    actors = [RemoteOptimise.remote(model, csv_path, sample_size, dt) for _ in range(num_jobs)]

    epochs = 2
    results = ray.get([a.train_loop.remote(epochs) for a in actors])

    time.sleep(3)  # Print is cut short without sleep

    ray.shutdown()









