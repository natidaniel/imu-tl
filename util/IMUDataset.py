from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging


def split_train_val(dataset, p=0.2):
    val_indices = []
    train_indices = list(range(len(dataset)))
    for cls in range(dataset.n_classes):
        # Split per class
        cls_indices = np.where(dataset.labels == cls)[0]
        n = len(cls_indices)
        # select indices from the class
        selected_indices = np.random.choice(n, size=int(p * n))
        # take validation
        selected = np.zeros(n).astype(np.bool)
        selected[selected_indices] = True
        val_indices += list(cls_indices[selected])
        # take train
        selected = np.ones(n).astype(np.bool)
        selected[selected_indices] = False
        train_indices += list(cls_indices[selected])

    return train_indices, val_indices


class IMUDataset(Dataset):
    """
        A class representing a dataset for IMU learning tasks
    """
    def __init__(self, imu_dataset_file, window_size, input_size,
                 window_shift=None, header=0, selected_indices=None, use_gps=False):
        """
        :param imu_dataset_file: (str) a file with imu signals and their labels
        :param window_size (int): the window size to consider
        :param input_size (int): the input size (e.g. 6 for 6 IMU measurements)
        :param window_shift (int): the overlap between each window
        :return: an instance of the class
        """
        super(IMUDataset, self).__init__()
        if window_shift is None:
            window_shift = window_size
        df = pd.read_csv(imu_dataset_file, header=header)
        if df.shape[1] == 1:
            df = pd.read_csv(imu_dataset_file, header=header, delimiter='\t')
        if selected_indices is not None:
            df = df.iloc[selected_indices, :]
        # Fetch the flatten IMU data and labels
        self.imu = df.iloc[:, :input_size].values
        self.raw_labels = df.iloc[:, -1].values
        n = self.raw_labels.shape[0]
        self.start_indices = list(range(0, n - window_size + 1, window_shift))
        self.window_size = window_size
        if use_gps:
            self.label_dict = {'1': 'Walk',
                               '2': 'Bike',
                               '3': 'Vehicle',
                               '4': 'Car',
                               '5': 'Motocycle'}
            # move 3: Bus
        else:  # IMU
            self.label_dict = {'1':'Jogging',
                              '2':'Sitting',
                              '3':'Stairsdown',
                              '4':'Stairsup',
                              '5':'Walking',
                              '6':'Stationary',
                              '7':'Biking',
                              '8':'Lying',
                              '9':'Running',
                              '10':'Cycling',
                              '11':'NordicWalking',
                              '12':'VacuumCleaning',
                              '13':'Ironing',
                              '14':'RopeJumping',
                              '15':'WatchingTv',
                              '16':'ComputerWork',
                              '17':'CarDriving',
                              '18':'FoldingLaundry',
                              '19':'HouseCleaning',
                              '20':'PlayingSoccer'}

        self.sorted_unique_raw_labels = np.sort(np.unique(self.raw_labels))
        self.n_classes = len(self.sorted_unique_raw_labels)
        raw_label_to_label_dict = dict(zip(self.sorted_unique_raw_labels, list(range(self.n_classes))))

        logging.info("Found {} unique classes,".format(self.n_classes))
        for i, l in enumerate(self.sorted_unique_raw_labels):
            logging.info("Class {} [id {}] mapped to id {}".format(self.label_dict.get(str(l)), l, i))

        self.labels = np.zeros(len(self.raw_labels)).astype(np.int)
        for i, l in enumerate(self.raw_labels):
            self.labels[i] = raw_label_to_label_dict.get(l)

        logging.info(
            "Number of windows: {} (generated from {} windows of size {} with shift {})".format(len(self.start_indices),
                                                                                                n // window_size,
                                                                                                window_size,
                                                                                                window_shift))

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        window_indices = list(range(start_index, (start_index + self.window_size)))
        imu = self.imu[window_indices, :]
        window_labels = self.labels[window_indices]
        #if len(np.unique(window_labels)) > 1:
        #    logging.warning("Window includes more than one class present, introducing noise")
        label = window_labels[0]
        sample = {'imu': imu,
                  'label': label,
                  'raw_label': self.sorted_unique_raw_labels[label]}
        return sample



