import os
from typing import Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np


class PatientDataset(Dataset):
    def __init__(
        self,
        features_path,
        dataset_path,
        patient: Optional[str] = None,  # if None then use data of all patients
        mode: str = "train",
        scaler=None,
        window_size: int = 48,
        stride: int = 12,
    ):
        self.features_path = features_path
        self.dataset_path = dataset_path  # to load relapses
        self.mode = mode
        self.window_size = window_size
        self.stride = stride

        # ---- Features used as input ----
        self.columns_to_scale = [
            "acc_norm", "gyr_norm", "heartRate_mean", "rRInterval_mean",
            "rRInterval_rmssd", "rRInterval_sdnn",
            "rRInterval_lombscargle_power_high", "steps"
        ]
        self.data_columns = self.columns_to_scale

        # ---- Features to predict (targets) ----
        self.target_columns = [
            "heartRate_mean",
            "rRInterval_mean",
            "rRInterval_rmssd",
            "rRInterval_sdnn",
            "rRInterval_lombscargle_power_high",
        ]

        self.data = []
        all_data = pd.DataFrame()

        # ---- Load data ----
        if patient is None:  # use all patients
            for patient in sorted(os.listdir(features_path)):
                if patient == ".DS_Store":
                    continue
                all_data = self.create_data(patient, mode, all_data)
        else:
            all_data = self.create_data(patient, mode, all_data)

        # ---- Fit or use existing scaler ----
        if scaler is None:
            self.scaler = MinMaxScaler()
            self.scaler.fit(all_data[self.columns_to_scale].dropna().to_numpy())
        else:
            self.scaler = scaler

        print(f"Created dataset `{mode}` for Patient {patient} of size: {len(self)}")

    def create_data(self, patient: str, mode: str, all_data: pd.DataFrame):
        patient_dir = os.path.join(self.features_path, patient)
        for subfolder in os.listdir(patient_dir):
            if (
                ("train" in mode and "train" in subfolder and subfolder.endswith("train"))
                or (mode == "val" and "val" in subfolder and subfolder.endswith("val"))
                or (mode == "test" and "test" in subfolder)
            ):
                subfolder_dir = os.path.join(patient_dir, subfolder)
                for file in os.listdir(subfolder_dir):
                    if file.endswith("features_stretched_w_steps.csv"):
                        file_path = os.path.join(subfolder_dir, file)
                        df = pd.read_csv(file_path)
                        df = df.replace([np.inf, -np.inf], np.nan)
                        df = df.dropna()

                        # Just verify all target columns exist
                        missing_targets = [c for c in self.target_columns if c not in df.columns]
                        if missing_targets:
                            print(f"Warning: Missing target columns {missing_targets} in {file_path}, skipping file.")
                            continue

                        all_data = pd.concat([all_data, df])

                        day_indices = df["day"].unique()

                        relapse_df = None
                        if "train" not in mode:
                            relapse_data_path = os.path.join(
                                self.features_path, patient, subfolder, "relapse_stretched.csv"
                            )
                            relapse_df = pd.read_csv(relapse_data_path)

                        for day_index in day_indices:
                            day_data = df[df["day"] == day_index].copy()

                            relapse_label = 0
                            if "train" not in mode and relapse_df is not None:
                                try:
                                    relapse_label = relapse_df[relapse_df["day"] == day_index]["relapse"].values[0]
                                except Exception:
                                    relapse_label = 0

                            if len(day_data) < self.window_size:
                                continue

                            if mode == "train":
                                # Overlapping sliding windows
                                for start_idx in range(0, len(day_data) - self.window_size, self.stride):
                                    sequence = day_data.iloc[start_idx:start_idx + self.window_size]
                                    sequence = sequence[self.data_columns].copy().to_numpy()

                                    # predict last timestep heart features
                                    slc = slice(start_idx + self.window_size - 1, start_idx + self.window_size)
                                    target = day_data.iloc[slc][self.target_columns].copy().to_numpy()

                                    self.data.append((sequence, target, int(patient[1:]), relapse_label))
                            else:
                                # Full-day sequences for val/test
                                self.data.append((day_data, None, int(patient[1:]), relapse_label))
        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        day_data, target, patient_id, relapse_label = self.data[idx]

        if self.mode == "train":
            sequence = self.scaler.transform(day_data)
            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).permute(1, 0)

            target_tensor = torch.tensor(target, dtype=torch.float32).permute(1, 0)

        else:  # val/test
            sequences, targets = [], []
            if len(day_data) < self.window_size:
                return None

            if len(day_data) == self.window_size:
                sequence = day_data.iloc[0:self.window_size][self.data_columns].copy().to_numpy()
                sequence = self.scaler.transform(sequence)
                sequences.append(sequence)

                tar = day_data.iloc[0:self.window_size][self.target_columns].copy().to_numpy()
                targets.append(tar)
            else:
                for start_idx in range(0, len(day_data) - self.window_size, self.window_size // 3):
                    sequence = day_data.iloc[start_idx:start_idx + self.window_size][self.data_columns].copy().to_numpy()
                    sequence = self.scaler.transform(sequence)
                    sequences.append(sequence)

                    target = day_data.iloc[start_idx + self.window_size - 1][self.target_columns].copy().to_numpy()
                    targets.append(target)

            sequence = np.stack(sequences)
            target = np.stack(targets)

            sequence_tensor = torch.tensor(sequence, dtype=torch.float32).permute(0, 2, 1)
            target_tensor = torch.tensor(target, dtype=torch.float32)

        return {
            "data": sequence_tensor,
            "target": target_tensor,  # now heart features
            "user_id": torch.tensor(patient_id, dtype=torch.long) - 1,
            "relapse_label": torch.tensor(relapse_label, dtype=torch.long),  # still used for evaluation
            "idx": idx,
        }
