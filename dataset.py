import os
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np


def calculate_sincos_from_minutes(minutes):
    """Convert minutes into cyclical sin/cos encoding"""
    time_value = minutes * (2. * np.pi / (60 * 24))
    sin_t = np.sin(time_value)
    cos_t = np.cos(time_value)
    return sin_t, cos_t


class PatientDataset(Dataset):
    def __init__(
            self,
            features_path,
            dataset_path,
            patient: Optional[str] = None,
            mode='train',
            scaler=None,
            window_size=48,
            stride=12,
    ):
        self.features_path = features_path
        self.dataset_path = dataset_path
        self.mode = mode
        self.window_size = window_size
        self.stride = stride

        # columns we normalize
        self.columns_to_scale = [
            'acc_norm', 'gyr_norm', 'heartRate_mean', 'rRInterval_mean',
            'rRInterval_rmssd', 'rRInterval_sdnn',
            'rRInterval_lombscargle_power_high', 'steps'
        ]
        self.data_columns = self.columns_to_scale

        # --- new: targets for circadian and sleep ---
        self.target_time_columns = ['sin_t', 'cos_t']
        self.target_sleep_columns = ['sin_onset', 'cos_onset', 'sin_wake', 'cos_wake']

        self.data = []
        all_data = pd.DataFrame()

        if patient is None:
            for patient in sorted(os.listdir(features_path)):
                if patient == ".DS_Store":
                    continue
                all_data = self.create_data(patient, mode, all_data)
        else:
            all_data = self.create_data(patient, mode, all_data)

        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(all_data[self.columns_to_scale].dropna().to_numpy())
        else:
            self.scaler = scaler

        print(f"Created dataset `{mode}`\tfor Patient {patient} of size: {len(self)}")

    def create_data(self, patient: str, mode: str, all_data: pd.DataFrame):
        patient_dir = os.path.join(self.features_path, patient)

        # try to load sleep features (day-level file)
        sleep_path = os.path.join(patient_dir, "sleep_features.csv")
        sleep_df = None
        if os.path.exists(sleep_path):
            sleep_df = pd.read_csv(sleep_path)

        for subfolder in os.listdir(patient_dir):
            if ('train' in mode and 'train' in subfolder and subfolder.endswith('train')) or \
                    (mode == 'val' and 'val' in subfolder and subfolder.endswith('val')) or \
                    (mode == 'test' and 'test' in subfolder):

                subfolder_dir = os.path.join(patient_dir, subfolder)
                for file in os.listdir(subfolder_dir):
                    if file.endswith('features_stretched_w_steps.csv'):
                        file_path = os.path.join(subfolder_dir, file)
                        df = pd.read_csv(file_path)
                        df = df.replace([np.inf, -np.inf], np.nan)
                        df = df.dropna()

                        # ensure time encodings exist
                        if not set(self.target_time_columns).issubset(df.columns):
                            mins = df['mins']
                            sin_t, cos_t = calculate_sincos_from_minutes(mins)
                            df['sin_t'] = sin_t
                            df['cos_t'] = cos_t

                        all_data = pd.concat([all_data, df])

                        day_indices = df['day'].unique()

                        if "train" not in mode:
                            relapse_data_path = os.path.join(
                                self.features_path, patient, subfolder, 'relapse_stretched.csv'
                            )
                            relapse_df = pd.read_csv(relapse_data_path)

                        for day_index in day_indices:
                            day_data = df[df['day'] == day_index].copy()

                            # default relapse label
                            relapse_label = 0
                            if "train" not in mode:
                                try:
                                    relapse_label = relapse_df[
                                        relapse_df['day'] == day_index
                                    ]['relapse'].values[0]
                                except:
                                    relapse_label = 0

                            # --- new: match with sleep features ---
                            sleep_target = np.zeros(4)
                            if sleep_df is not None:
                                match = sleep_df[sleep_df['day_index'] == day_index]
                                if not match.empty:
                                    sleep_target = match[self.target_sleep_columns].iloc[0].to_numpy()

                            if len(day_data) < self.window_size:
                                continue

                            if mode == "train":
                                for start_idx in range(0, len(day_data) - self.window_size, self.stride):
                                    sequence = day_data.iloc[start_idx:start_idx+self.window_size]
                                    sequence = sequence[self.data_columns].copy().to_numpy()

                                    # target time = last timestep
                                    slc = slice(start_idx + self.window_size - 1, start_idx + self.window_size)
                                    target_time = day_data.iloc[slc][self.target_time_columns].copy().to_numpy()

                                    self.data.append((
                                        sequence, target_time, sleep_target, int(patient[1:]), relapse_label
                                    ))
                            else:
                                self.data.append((day_data, None, sleep_target, int(patient[1:]), relapse_label))
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        day_data, target_time, target_sleep, patient_id, relapse_label = self.data[idx]

        if self.mode == 'train':
            sequence = self.scaler.transform(day_data)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
            sequence_tensor = sequence_tensor.permute(1, 0)

            target_time_tensor = torch.tensor(target_time, dtype=torch.float32).squeeze(0)
            target_sleep_tensor = torch.tensor(target_sleep, dtype=torch.float32)

        else:
            sequences, targets_time = [], []
            if len(day_data) < self.window_size:
                return None

            if len(day_data) == self.window_size:
                sequence = day_data.iloc[0:self.window_size][self.data_columns].copy().to_numpy()
                sequence = self.scaler.transform(sequence)
                sequences.append(sequence)

                tar = day_data.iloc[0:self.window_size][self.target_time_columns].copy().to_numpy()
                targets_time.append(tar)
            else:
                for start_idx in range(0, len(day_data) - self.window_size, self.window_size//3):
                    sequence = day_data.iloc[start_idx:start_idx+self.window_size][self.data_columns].copy().to_numpy()
                    sequence = self.scaler.transform(sequence)
                    sequences.append(sequence)

                    target_time = day_data.iloc[start_idx + self.window_size - 1][self.target_time_columns].copy().to_numpy()
                    targets_time.append(target_time)

            sequence = np.stack(sequences)
            target_time = np.stack(targets_time)

            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).permute(0, 2, 1)
            target_time_tensor = torch.tensor(target_time, dtype=torch.float32)
            target_sleep_tensor = torch.tensor(target_sleep, dtype=torch.float32)

        return {
            'data': sequence_tensor,
            'target_time': target_time_tensor,
            'target_sleep': target_sleep_tensor,
            'user_id': torch.tensor(patient_id, dtype=torch.long)-1,
            'relapse_label': torch.tensor(relapse_label, dtype=torch.long),
            'idx': idx,
        }
