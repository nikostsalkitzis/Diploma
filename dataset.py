import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class PatientDataset(Dataset):
    def __init__(self, features_path, dataset_path, mode='train', scaler=None, window_size=48):
        self.features_path = features_path
        self.dataset_path = dataset_path
        self.mode = mode
        self.window_size = window_size
        
        # columns to scale for CNN input
        self.columns_to_scale = [
            'acc_norm', 'heartRate_mean', 'rRInterval_mean', 
            'rRInterval_rmssd', 'rRInterval_sdnn', 'rRInterval_lombscargle_power_high'
        ]
        self.data_columns = self.columns_to_scale + ['sin_t', 'cos_t']

        self.data = []

        all_data = pd.DataFrame()

        # iterate patients
        for patient in sorted(os.listdir(features_path)):
            patient_dir = os.path.join(features_path, patient)
            for subfolder in os.listdir(patient_dir):
                if (mode == 'train' and 'train' in subfolder) or (mode == 'val' and 'val' in subfolder) or (mode == 'test' and 'test' in subfolder):
                    subfolder_dir = os.path.join(patient_dir, subfolder)
                    for file in os.listdir(subfolder_dir):
                        if file.endswith('.csv'):
                            file_path = os.path.join(subfolder_dir, file)
                            df = pd.read_csv(file_path, index_col=0)
                            df = df.replace([np.inf, -np.inf], np.nan).dropna()
                            all_data = pd.concat([all_data, df])
                            
                            # load relapse labels
                            relapse_df = pd.read_csv(os.path.join(self.dataset_path, patient, subfolder, 'relapses.csv'))
                            
                            for day_index in df['day_index'].unique():
                                day_data = df[df['day_index']==day_index].copy()
                                relapse_label = relapse_df[relapse_df['day_index']==day_index]['relapse'].values[0]

                                if len(day_data) < self.window_size:
                                    continue

                                if mode == 'train':
                                    # sliding window with 1-hour overlap (12 steps)
                                    for start_idx in range(0, len(day_data) - self.window_size, 12):
                                        sequence = day_data.iloc[start_idx:start_idx+self.window_size][self.data_columns].to_numpy()
                                        self.data.append((sequence, int(patient[1:])-1, relapse_label))
                                else:
                                    self.data.append((day_data, int(patient[1:])-1, relapse_label))

        if scaler is None:
            print(mode, "fitting scaler")
            self.scaler = MinMaxScaler()
            self.scaler.fit(all_data[self.columns_to_scale].to_numpy())
        else:
            self.scaler = scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        day_data, patient_id, relapse_label = self.data[idx]

        # train mode: scale, convert to tensor, channels-first for CNN
        if self.mode == 'train':
            sequence = day_data.copy()
            sequence[:, :-2] = self.scaler.transform(sequence[:, :-2])  # scale only non-time columns
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).permute(1, 0)  # (channels, seq_len)
        else:
            # validation/test: handle multiple overlapping sequences per day
            sequences = []
            day_data = day_data[self.data_columns].to_numpy()
            if len(day_data) < self.window_size:
                return None
            elif len(day_data) == self.window_size:
                sequences.append(day_data)
            else:
                for start_idx in range(0, len(day_data) - self.window_size, self.window_size // 3):
                    seq = day_data[start_idx:start_idx+self.window_size]
                    sequences.append(seq)
            sequences = np.stack(sequences)
            sequences[:, :, :-2] = self.scaler.transform(sequences[:, :, :-2].reshape(-1, len(self.columns_to_scale))).reshape(sequences.shape[0], sequences.shape[1], len(self.columns_to_scale))
            sequence_tensor = torch.tensor(sequences, dtype=torch.float32).permute(0, 2, 1)  # (num_seq, channels, seq_len)

        return {
            'data': sequence_tensor,
            'user_id': torch.tensor(patient_id, dtype=torch.long),
            'relapse_label': torch.tensor(relapse_label, dtype=torch.long)
        }
