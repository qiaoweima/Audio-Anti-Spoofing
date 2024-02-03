import pandas as pd
import torch
from torch.utils.data.dataloader import Dataset
import soundfile as sf


class PrepASV19FullDataset(Dataset):
    def __init__(self, protocol_file_path, cqt_data_path, lfcc_data_path, logspec_data_path):
        self.train_protocol = pd.read_csv(protocol_file_path, sep=' ', header=None)
        self.cqt_data_path = cqt_data_path
        self.lfcc_data_path = lfcc_data_path
        self.logspec_data_path = logspec_data_path

    def __len__(self):
        return self.train_protocol.shape[0]

    def __getitem__(self, index):
        cqt_data_file_path = self.cqt_data_path + self.train_protocol.iloc[index, 1]
        lfcc_data_file_path = self.lfcc_data_path + self.train_protocol.iloc[index, 1]
        logspec_data_file_path = self.logspec_data_path + self.train_protocol.iloc[index, 1]

        cqt_sample = torch.load(cqt_data_file_path + '.pt')
        lfcc_sample = torch.load(lfcc_data_file_path + '.pt')
        logspec_sample = torch.load(logspec_data_file_path + '.pt')
        cqt_sample = torch.tensor(cqt_sample, dtype=torch.float32)
        cqt_sample = torch.unsqueeze(cqt_sample, 0)
        lfcc_sample = torch.tensor(lfcc_sample, dtype=torch.float32)
        lfcc_sample = torch.unsqueeze(lfcc_sample, 0)
        logspec_sample = torch.tensor(logspec_sample, dtype=torch.float32)
        logspec_sample = torch.unsqueeze(logspec_sample, 0)
        label = self.train_protocol.iloc[index, 4]
        label = label_encode(label)
        sub_class = self.train_protocol.iloc[index, 3]
        sub_class = sub_class_encode_19(sub_class)
        return cqt_sample, lfcc_sample, logspec_sample, label, sub_class

    def get_weights(self):
        label_info = self.train_protocol.iloc[:, 4]
        num_zero_class = (label_info == 'bonafide').sum()
        num_one_class = (label_info == 'spoof').sum()
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights

class PrepASV19Dataset(Dataset):
    def __init__(self, protocol_file_path, data_path, data_type='time_frame'):
        self.train_protocol = pd.read_csv(protocol_file_path, sep=' ', header=None)
        self.data_path = data_path
        self.data_type = data_type

    def __len__(self):
        return self.train_protocol.shape[0]

    def __getitem__(self, index):
        data_file_path = self.data_path + self.train_protocol.iloc[index, 1]

        if self.data_type == 'time_frame':
            sample, _ = sf.read(data_file_path + '.flac')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            meta_label = self.train_protocol.iloc[index, 4]
            label = label_encode(meta_label)
            # sub_class = self.train_protocol.iloc[index, 3]
            # sub_class = sub_class_encode_19(sub_class)
            return sample, label, meta_label

        if self.data_type == 'wav2clip':
            sample = torch.load(data_file_path + '.pt')
            sample = torch.tensor(sample, dtype=torch.float32)
            # sample = torch.unsqueeze(sample, 0)
            meta_label = self.train_protocol.iloc[index, 4]
            label = label_encode(meta_label)
            # sub_class = self.train_protocol.iloc[index, 3]
            # sub_class = sub_class_encode_19(sub_class)
            return sample, label, meta_label

        if self.data_type == 'CQT':
            sample = torch.load(data_file_path + '.pt')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 3]
            sub_class = sub_class_encode_19(sub_class)
            return sample, label, sub_class

        if self.data_type == 'LogSpec':
            sample = torch.load(data_file_path + '.pt')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 3]
            sub_class = sub_class_encode_19(sub_class)
            return sample, label, sub_class
        
        if self.data_type == 'LFCC':
            sample = torch.load(data_file_path + '.pt')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 3]
            sub_class = sub_class_encode_19(sub_class)
            return sample, label, sub_class

        if self.data_type == 'CQCC':
            sample = torch.load(data_file_path + '.pt')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 3]
            sub_class = sub_class_encode_19(sub_class)
            return sample, label, sub_class

        if self.data_type == 'MFCC':
            sample = torch.load(data_file_path + '.pt')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 3]
            sub_class = sub_class_encode_19(sub_class)
            return sample, label, sub_class

    def get_weights(self):
        label_info = self.train_protocol.iloc[:, 4]
        num_zero_class = (label_info == 'bonafide').sum()
        num_one_class = (label_info == 'spoof').sum()
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights


class PrepASV21Dataset(Dataset):
    def __init__(self, protocol_file_path, data_path, data_type='time_frame'):
        self.train_protocol = pd.read_csv(protocol_file_path, sep=' ', header=None)
        # self.forin = self.train_protocol.index[self.train_protocol[7] == 'eval'].tolist()
        # self.train_protocol = self.train_protocol.values
        # self.tra = self.train_protocol[self.forin,:]

        self.data_path = data_path
        self.data_type = data_type

    def __len__(self):
        return self.train_protocol.shape[0]

    def __getitem__(self, index):
        # data_file_path = self.data_path + self.tra[index, 1]
        data_file_path = self.data_path + self.train_protocol.iloc[index, 1]

        if self.data_type == 'time_frame':
            sample, _ = sf.read(data_file_path + '.flac')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            meta_label = self.train_protocol.iloc[index, 5]
            label = label_encode(meta_label)
            # sub_class = self.train_protocol.iloc[index, 3]
            # sub_class = sub_class_encode_19(sub_class)
            return sample, label, meta_label


    def get_weights(self):
        label_info = self.train_protocol.iloc[:, 5]
        num_zero_class = (label_info == 'bonafide').sum()
        num_one_class = (label_info == 'spoof').sum()
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights


class PrepASV15Dataset(Dataset):
    def __init__(self, protocol_file_path, data_path, data_type='time_frame'):
        self.train_protocol = pd.read_csv(protocol_file_path, sep=' ', header=None)
        self.data_path = data_path
        self.data_type = data_type

    def __len__(self):
        return self.train_protocol.shape[0]

    def __getitem__(self, index):
        data_file_path = self.data_path + self.train_protocol.iloc[index, 1]

        if self.data_type == 'time_frame':
            sample, _ = sf.read(data_file_path + '.wav')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 3]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 2]
            sub_class = sub_class_encode_15(sub_class)
            return sample, label, sub_class

        if self.data_type == 'CQT':
            sample = torch.load(data_file_path + '.pt')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 3]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 2]
            sub_class = sub_class_encode_15(sub_class)
            return sample, label, sub_class

    def get_weights(self):
        label_info = self.train_protocol.iloc[:, 3]
        num_zero_class = (label_info == 'human').sum()
        num_one_class = (label_info == 'spoof').sum()
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights


def label_encode(label):
    if label == 'bonafide':
        label = torch.tensor(0, dtype=torch.int64)
    elif label == 'human':
        label = torch.tensor(0, dtype=torch.int64)
    else:
        label = torch.tensor(1, dtype=torch.int64)
    return label


def sub_class_encode_19(label):
    if label == '-':
        label = torch.tensor(0, dtype=torch.int64)
    elif label == 'A01':
        label = torch.tensor(1, dtype=torch.int64)
    elif label == 'A02':
        label = torch.tensor(2, dtype=torch.int64)
    elif label == 'A03':
        label = torch.tensor(3, dtype=torch.int64)
    elif label == 'A04':
        label = torch.tensor(4, dtype=torch.int64)
    elif label == 'A05':
        label = torch.tensor(5, dtype=torch.int64)
    elif label == 'A06':
        label = torch.tensor(6, dtype=torch.int64)
    elif label == 'A07':
        label = torch.tensor(7, dtype=torch.int64)
    elif label == 'A08':
        label = torch.tensor(8, dtype=torch.int64)
    elif label == 'A09':
        label = torch.tensor(9, dtype=torch.int64)
    elif label == 'A10':
        label = torch.tensor(10, dtype=torch.int64)
    elif label == 'A11':
        label = torch.tensor(11, dtype=torch.int64)
    elif label == 'A12':
        label = torch.tensor(12, dtype=torch.int64)
    elif label == 'A13':
        label = torch.tensor(13, dtype=torch.int64)
    elif label == 'A14':
        label = torch.tensor(14, dtype=torch.int64)
    elif label == 'A15':
        label = torch.tensor(15, dtype=torch.int64)
    elif label == 'A16':
        label = torch.tensor(16, dtype=torch.int64)
    elif label == 'A17':
        label = torch.tensor(17, dtype=torch.int64)
    elif label == 'A18':
        label = torch.tensor(18, dtype=torch.int64)
    elif label == 'A19':
        label = torch.tensor(19, dtype=torch.int64)
    return label


def sub_class_encode_15(label):
    if label == 'human':
        label = torch.tensor(0, dtype=torch.int64)
    elif label == 'S1':
        label = torch.tensor(1, dtype=torch.int64)
    elif label == 'S2':
        label = torch.tensor(2, dtype=torch.int64)
    elif label == 'S3':
        label = torch.tensor(3, dtype=torch.int64)
    elif label == 'S4':
        label = torch.tensor(4, dtype=torch.int64)
    elif label == 'S5':
        label = torch.tensor(5, dtype=torch.int64)
    elif label == 'S6':
        label = torch.tensor(6, dtype=torch.int64)
    elif label == 'S7':
        label = torch.tensor(7, dtype=torch.int64)
    elif label == 'S8':
        label = torch.tensor(8, dtype=torch.int64)
    elif label == 'S9':
        label = torch.tensor(9, dtype=torch.int64)
    elif label == 'S10':
        label = torch.tensor(10, dtype=torch.int64)
    return label
