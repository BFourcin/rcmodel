from torch.utils.data import Dataset
from random import randint
import torch
import pandas as pd


class BuildingTemperatureDataset(Dataset):
    """
    Splits dataset up into batches of len(dataset) // sample_size. Note remainder of data is thrown away.
    train and test tags can be used to select a percentage slice of data in this order, see function _split_dataset() for current % splits.

    If there is insufficient data for one batch, sample_size will be reduced to match the data.
    """

    def __init__(self, csv_path, sample_size, transform=None, all=True, train=False, test=False):
        self.csv_path = csv_path
        self.transform = transform
        self.sample_size = int(sample_size)  # this is the number of rows of data from the csv per sample.
        self.headings = list(pd.read_csv(csv_path, nrows=1))  # list of dataframe headings
        self.all = all
        self.train = train
        self.test = test

        # auto splits data by train and test
        # entry count is number of rows to read from csv.
        # remainder of data e.g. after entry_count//sample_size is lost
        self.rows_to_skip, self.entry_count = self._split_dataset()

        self.__len__()  # Used to initialise logic used in __len__.

    def __len__(self):
        """
        Get number of batches in the dataset. Returns int
        Minimum of 1 batch will be returned.
        """

        num_samples = self.entry_count // self.sample_size

        # Insufficient data for 1 whole sample size. Remainder of data used instead.
        if num_samples == 0:
            num_samples = 1
            self.sample_size = self.entry_count  # Reduce sample to the entries remaining

        return num_samples

    def __getitem__(self, idx):
        """
        Returns a 'sample' of the total dataset.

        Each sample has length 'sample_size' except for the final one, which contains the remainder of the dataset.

        The entire dataset is never loaded in at once, each sample is read in separately when needed to reduce memory usage.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get upper and lower bounds of dataset slice
        lb = idx * self.sample_size + self.rows_to_skip
        ub = (idx + 1) * self.sample_size + self.rows_to_skip

        # Get pandas df of sample
        df_sample = pd.read_csv(self.csv_path, skiprows=lb, nrows=self.sample_size)

        # Get time column (time must be in the 1th column)
        t_sample = torch.tensor(df_sample.iloc[:, 1].values, dtype=torch.float64)  # units (s)

        # Get temp matrix
        temp_sample = torch.tensor(df_sample.iloc[:, 2:].values,
                                   dtype=torch.float32)  # pandas needs 2: to get all but first & second column

        # apply transforms if required
        if self.transform:
            temp_sample = self.transform(temp_sample)

        return t_sample, temp_sample

    def _get_entries(self):
        """Get total rows/entries of data in csv"""
        from csv import reader

        # the rows in the .csv are counted:
        with open(self.csv_path, "r") as f:
            read_f = reader(f, delimiter=",")
            entry_count = sum(1 for row in read_f) - 1  # minus one to account for heading

        return entry_count

    def _split_dataset(self):

        train_split = 0.8
        test_split = 0.2

        total_entries = self._get_entries()  # total rows of data in csv

        if self.train:
            rows_to_skip = 0
            entry_count = int(total_entries * train_split)
            return rows_to_skip, entry_count
        elif self.test:
            rows_to_skip = int(total_entries * train_split)
            entry_count = int(total_entries * test_split)
            return rows_to_skip, entry_count
        elif self.all:
            rows_to_skip = 0
            entry_count = total_entries
            return rows_to_skip, entry_count
        else:
            raise ValueError('train, test and validation all False')

    def get_all_data(self):
        # Get upper and lower bounds of dataset slice
        lb = self.rows_to_skip

        # Get pandas df of sample
        df_sample = pd.read_csv(self.csv_path, skiprows=lb, nrows=self.entry_count)

        # Get time column (time must be in the 1th column)
        t_sample = torch.tensor(df_sample.iloc[:, 1].values, dtype=torch.float64)  # units (s)

        # Get temp matrix
        temp_sample = torch.tensor(df_sample.iloc[:, 2:].values,
                                   dtype=torch.float32)  # pandas needs 2: to get all but first & second column

        # apply transforms if required
        if self.transform:
            temp_sample = self.transform(temp_sample)

        return t_sample, temp_sample


class RandomSampleDataset(Dataset):
    """
    Used to return random windows of data from a dataset.

    The initial part of the data (len=warmup_size) is reserved to be used to "warm-up" the latent variables of the
    model and is not returned by this class.

    Order of the data set goes: warm-up data, training data, testing data with no overlaps.
    len(dataset_train) = (len(data_set)-warmup_size) * 0.8
    len(dataset_test) = (len(data_set)-warmup_size) * 0.2
    """

    def __init__(self, csv_path, sample_size, warmup_size, transform=None, all=True, train=False, test=False):
        self.csv_path = csv_path
        self.transform = transform
        self.sample_size = int(sample_size)  # this is the number of rows of data from the csv per sample.
        self.warmup_size = int(warmup_size)  # amount of warmup data used to get a better result for iv.
        self.headings = list(pd.read_csv(csv_path, nrows=1))  # list of dataframe headings
        self.all = all
        self.train = train
        self.test = test

        # auto splits data by train and test
        # entry count is total number of rows in the csv which belong in this dataset.
        self.rows_to_skip, self.entry_count = self._split_dataset()

        self.__len__()  # Used to initialise logic used in __len__.

    def __len__(self):
        """
        Get number of batches in the dataset. Returns int
        Minimum of 1 batch will be returned.
        """

        num_samples = self.entry_count // self.sample_size

        # Insufficient data for 1 whole sample size. Remainder of data used instead.
        if num_samples == 0:
            raise ValueError('Insufficient amount of data')

        return num_samples

    def __getitem__(self, idx=0):
        """
        Returns a 'sample' of the total dataset.

        A window of data is randomly selected from the data remaining after we have removed what was used in the warm up.

        The entire dataset is never loaded in at once, each sample is read in separately when needed to reduce memory usage.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # we take a random index from the range of valid indexes:
        start_idx = randint(0, self.entry_count - self.sample_size) + self.rows_to_skip

        # Get pandas df of sample
        df_sample = pd.read_csv(self.csv_path, skiprows=start_idx, nrows=self.sample_size)

        # Get time column (time must be in the 1th column)
        t_sample = torch.tensor(df_sample.iloc[:, 1].values, dtype=torch.float64)  # units (s)

        # Get temp matrix
        temp_sample = torch.tensor(df_sample.iloc[:, 2:].values,
                                   dtype=torch.float32)  # pandas needs 2: to get all but first & second column

        # apply transforms if required
        if self.transform:
            temp_sample = self.transform(temp_sample)

        return t_sample, temp_sample

    def _get_entries(self):
        """Get total rows/entries of data in csv"""
        from csv import reader

        # the rows in the .csv are counted:
        with open(self.csv_path, "r") as f:
            read_f = reader(f, delimiter=",")
            total_entries = sum(1 for row in read_f) - 1  # minus one to account for heading
            
        total_entries -= self.warmup_size  # remove the chunk of data used soley for warm up

        # check there is sufficient data
        assert total_entries > 0 and total_entries > self.sample_size

        return total_entries

    def _split_dataset(self):

        train_split = 0.8
        test_split = 0.2

        total_entries = self._get_entries()  # total rows of data in csv

        if self.train:
            rows_to_skip = self.warmup_size
            entry_count = int(total_entries * train_split)
            return rows_to_skip, entry_count
        elif self.test:
            rows_to_skip = int(total_entries * train_split) - self.warmup_size  # do warm up in training section.
            entry_count = int(total_entries * test_split)
            return rows_to_skip, entry_count
        elif self.all:
            rows_to_skip = self.warmup_size
            entry_count = total_entries
            return rows_to_skip, entry_count
        else:
            raise ValueError('train, test and validation all False')

    def get_all_data(self):
        start_idx = self.rows_to_skip

        # Get pandas df of entire valid dataset (iterate to avoid errors)
        # df_sample = pd.read_csv(self.csv_path, skiprows=start_idx, nrows=self.entry_count + self.sample_size)  #OLD
        iter_csv = pd.read_csv(self.csv_path, skiprows=start_idx, nrows=self.entry_count + self.sample_size,
                               iterator=True, chunksize=10000)
        df_sample = pd.concat([chunk.dropna(how='all') for chunk in iter_csv])

        # Get time column (time must be in the 1th column)
        t_sample = torch.tensor(df_sample.iloc[:, 1].values, dtype=torch.float64)  # units (s)

        # Get temp matrix
        temp_sample = torch.tensor(df_sample.iloc[:, 2:].values, dtype=torch.float32)

        # apply transforms if required
        if self.transform:
            temp_sample = self.transform(temp_sample)

        return t_sample, temp_sample
