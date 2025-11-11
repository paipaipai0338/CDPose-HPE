import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import os
from utils import split_dataset_by_frame_num


class MyDataset(Dataset):
    def __init__(self, root_path, mode, seq_len=None, action=None):
        assert mode in ['train', 'val'], 'err1'
        self.seq_len = seq_len
        radar_path = root_path + '/Radar'
        imu_path = root_path + '/IMU'

        result = split_dataset_by_frame_num(imu_path)

        # self.radar_data : list [frame] array [num, 5]
        # self.imu_data : array [frame, 23, 3]
        if seq_len == None:
            self.radar_data = []
            self.imu_data = np.zeros((0, 23, 3))
            if action == None:
                action = os.listdir(imu_path)
            for ac in action:
                objects = result[ac][mode]
                for ob in objects:
                    files = os.listdir(imu_path + '/' + ac + '/' + ob)
                    for f in files:
                        data = np.load(imu_path + '/' + ac + '/' + ob + '/' + f)
                        root = data[0, 0, :]
                        root[2] = 0
                        data = data - root
                        self.imu_data = np.concatenate((self.imu_data, data), axis=0)
                        with open(radar_path + '/' + ac + '/' + ob + '/' + f.split('.')[0] + '.pkl', 'rb') as ff:
                            data = pickle.load(ff)
                            data_acc = []
                            for i in range(0, len(data)):
                                if i == 0:
                                    data_acc.append(np.concatenate((data[i], data[i + 1]), axis=0))
                                elif i == len(data) - 1:
                                    data_acc.append(np.concatenate((data[i - 1], data[i]), axis=0))
                                else:
                                    data_acc.append(np.concatenate((data[i - 1], data[i], data[i + 1]), axis=0))
                            self.radar_data = self.radar_data + data_acc
            valid_mask = [r.shape[0] > 30 for r in self.radar_data]
            valid_mask = np.array(valid_mask)
            self.radar_data = np.array(self.radar_data, dtype=object)[valid_mask]
            self.imu_data = self.imu_data[valid_mask]
        else:
            self.radar_data = []
            self.imu_data = []
            action = os.listdir(imu_path)
            for ac in action:
                objects = result[ac][mode]
                for ob in objects:
                    files = os.listdir(imu_path + '/' + ac + '/' + ob)
                    for f in files:
                        data_imu = np.load(imu_path + '/' + ac + '/' + ob + '/' + f)
                        root = data_imu[0, 0, :]
                        root[2] = 0
                        data_imu = data_imu - root
                        with open(radar_path + '/' + ac + '/' + ob + '/' + f.split('.')[0] + '.pkl', 'rb') as ff:
                            data_radar = pickle.load(ff)
                            valid_mask = [r.shape[0] > 5 for r in data_radar]
                            valid_mask = np.array(valid_mask)
                            data_radar = np.array(data_radar, dtype=object)[valid_mask]
                        for i in range(0, len(data_radar) - seq_len, seq_len):
                            data_radar_acc = []
                            data_imu_acc = np.zeros((1, seq_len, 23, 3))
                            for j in range(seq_len):
                                data_radar_acc.append(data_radar[i + j])
                                data_imu_acc[0, j] = data_imu[i + j]
                            self.radar_data.append(data_radar_acc)
                            self.imu_data.append(data_imu_acc)
            self.imu_data = np.concatenate(self.imu_data, axis=0)

    def __len__(self):
        return self.imu_data.shape[0]

    def __getitem__(self, item):
        radar_data = self.radar_data[item]  # array [num, 5] or list [seq_len] array [23, 3]
        imu_data = self.imu_data[item]  # array [23, 3] or array [seq_len, 23, 3]
        if self.seq_len is None:
            valid_position = np.ones(radar_data.shape[0], )
            zero_pad_length = 600 - radar_data.shape[0]
            radar_data = np.concatenate((radar_data, np.zeros((zero_pad_length, 5))), axis=0)
            valid_position = np.concatenate((valid_position, np.zeros((zero_pad_length,))), axis=0)
        else:
            radar_data_ = np.zeros((self.seq_len, 600, 5))
            imu_data_ = np.zeros((self.seq_len, 23, 3))
            valid_position = np.zeros((self.seq_len, 600))
            for i in range(self.seq_len):
                radar_data_i = radar_data[i]
                imu_data_i = imu_data[i]
                radar_root = np.mean(radar_data_i[:, 0:2], axis=0)
                imu_root = np.mean(imu_data_i[:, 0:2], axis=0)
                radar_data_i[:, 0:2] = radar_data_i[:, 0:2] - radar_root + imu_root
                valid_position_i = np.ones(radar_data_i.shape[0], )
                zero_pad_length = 600 - radar_data_i.shape[0]
                radar_data_i = np.concatenate((radar_data_i, np.zeros((zero_pad_length, 5))), axis=0)
                valid_position_i = np.concatenate((valid_position_i, np.zeros((zero_pad_length,))), axis=0)
                radar_data_[i] = radar_data_i
                valid_position[i] = valid_position_i
                imu_data_[i] = imu_data_i
            radar_data = radar_data_
            imu_data = imu_data_
        radar_data = np.nan_to_num(radar_data, nan=0.0)
        return torch.tensor(radar_data).float(), torch.tensor(valid_position).bool(), torch.tensor(imu_data).float()

