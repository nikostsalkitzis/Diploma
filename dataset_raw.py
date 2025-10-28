import os
import warnings
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import welch


# -----------------------------
# Constants / channel layout
# -----------------------------

VALID_RANGES = {
    "acc": (-19.6, 19.6),      # accelerometer X/Y/Z valid range
    "gyr": (-573.0, 573.0),    # gyroscope X/Y/Z valid range
    "hr":  (0.0, 255.0),       # heartRate bpm valid range
    "rr":  (0.0, 2000.0),      # rRInterval ms valid range
}

ACC_COLS_RAW = ["X", "Y", "Z"]
GYR_COLS_RAW = ["X", "Y", "Z"]
HRM_COLS_RAW = ["heartRate", "rRInterval"]

# Model channel order:
# accX, accY, accZ,
# gyrX, gyrY, gyrZ,
# heartRate, rRInterval,
# sin_t, cos_t
ALL_CHANNELS = [
    "acc_X", "acc_Y", "acc_Z",
    "gyr_X", "gyr_Y", "gyr_Z",
    "heartRate", "rRInterval",
    "sin_t", "cos_t",
]

# Each "5-minute bin":
# 5 minutes = 300 seconds
# HRM is 5 Hz -> 5 samples/sec
# samples_per_bin = 300 * 5 = 1500
SAMPLES_PER_BIN = 300 * 5  # 1500


# -----------------------------
# HRV / heart feature helpers
# -----------------------------

def _rmssd(rr: np.ndarray) -> float:
    rr = rr[~np.isnan(rr)]
    if rr.size < 3:
        return np.nan
    diff = np.diff(rr)
    return float(np.sqrt(np.mean(diff * diff)))


def _sdnn(rr: np.ndarray) -> float:
    rr = rr[~np.isnan(rr)]
    if rr.size < 2:
        return np.nan
    return float(np.std(rr, ddof=1))


def _hf_power(rr: np.ndarray, fs: float = 5.0) -> float:
    """
    High-frequency (0.15–0.40 Hz) HRV power using Welch PSD.
    rr is assumed to be evenly sampled rRInterval-like signal at fs Hz.
    """
    clean = rr[~np.isnan(rr)]
    if clean.size < int(fs * 30):  # need at least ~30 seconds of data
        return np.nan

    f, Pxx = welch(clean, fs=fs, nperseg=min(len(clean), 256))
    mask = (f >= 0.15) & (f <= 0.40)
    if not np.any(mask):
        return np.nan
    return float(np.trapz(Pxx[mask], f[mask]))


def _compute_window_targets(hr_win: np.ndarray,
                            rr_win: np.ndarray,
                            fs: float = 5.0) -> np.ndarray:
    """
    Returns (5,) vector of:
    [ mean(HR), mean(RR), RMSSD, SDNN, HF_power ]
    """
    hr_mean = float(np.nanmean(hr_win)) if hr_win.size > 0 else np.nan
    rr_mean = float(np.nanmean(rr_win)) if rr_win.size > 0 else np.nan
    rmssd_v = _rmssd(rr_win)
    sdnn_v  = _sdnn(rr_win)
    hf_pow  = _hf_power(rr_win, fs=fs)

    return np.array([hr_mean, rr_mean, rmssd_v, sdnn_v, hf_pow], dtype=np.float32)


# -----------------------------
# Processing helpers
# -----------------------------

def _downsample_20hz_to_5hz(x_20hz: torch.Tensor) -> torch.Tensor:
    """
    Convert ~20 Hz signal to ~5 Hz using avg pooling.
    x_20hz: (C, T20)
    return: (C, T5)
    """
    x = x_20hz.unsqueeze(0)  # (1,C,T20)
    x5 = F.avg_pool1d(x, kernel_size=4, stride=4)  # (1,C,T5)
    return x5.squeeze(0)  # (C,T5)


def _get_minutes_from_timestr_col(time_series: pd.Series) -> np.ndarray:
    """
    Convert the 'time' column into minutes-from-midnight as integers.

    We suppress the pandas "Could not infer format" warnings here,
    but still coerce each value to datetime so we support:
      - "HH:MM:SS"
      - "HH:MM:SS.sss"
      - "YYYY-MM-DD HH:MM:SS"
      - etc.
    """
    # We locally silence Pandas' fallback warning so logs stay clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        t = pd.to_datetime(time_series.astype(str), errors="coerce")

    hours = t.dt.hour.to_numpy()
    mins  = t.dt.minute.to_numpy()
    return hours * 60 + mins


def _impute_nans_channelwise(arr: np.ndarray) -> np.ndarray:
    """
    arr: (C, T)
    For each channel c:
      - interpolate NaNs
      - fallback to channel median
      - fallback to 0.0
    """
    C, T = arr.shape
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(C):
        s = pd.Series(arr[c, :])
        s_interp = s.interpolate(limit_direction="both")
        if s_interp.isna().any():
            med = np.nanmedian(arr[c, :])
            if np.isnan(med):
                med = 0.0
            s_interp = s_interp.fillna(med)
        out[c, :] = s_interp.to_numpy(dtype=np.float32)
    return out


def _build_day_tensor(
    df_acc_day: pd.DataFrame,
    df_gyr_day: pd.DataFrame,
    df_hrm_day: pd.DataFrame,
):
    """
    Build a single day's aligned 5 Hz multichannel signal.

    Steps:
    - clip invalid acc/gyr/hr values to NaN
    - downsample acc/gyr from ~20 Hz -> 5 Hz
    - align acc/gyr/hrm to common length T5
    - generate sin_t/cos_t time-of-day channels
    - return day_tensor (10, T5), hr_sig (T5,), rr_sig (T5,)
    """

    # 1. Range filter accel / gyro
    for col in ACC_COLS_RAW:
        bad = (
            (df_acc_day[col] < VALID_RANGES["acc"][0])
            | (df_acc_day[col] > VALID_RANGES["acc"][1])
        )
        df_acc_day.loc[bad, col] = np.nan

    for col in GYR_COLS_RAW:
        bad = (
            (df_gyr_day[col] < VALID_RANGES["gyr"][0])
            | (df_gyr_day[col] > VALID_RANGES["gyr"][1])
        )
        df_gyr_day.loc[bad, col] = np.nan

    # 2. Range filter heart / rRInterval
    bad_hr = (
        (df_hrm_day["heartRate"] <= VALID_RANGES["hr"][0])
        | (df_hrm_day["heartRate"] > VALID_RANGES["hr"][1])
    )
    df_hrm_day.loc[bad_hr, "heartRate"] = np.nan

    bad_rr = (
        (df_hrm_day["rRInterval"] <= VALID_RANGES["rr"][0])
        | (df_hrm_day["rRInterval"] > VALID_RANGES["rr"][1])
    )
    df_hrm_day.loc[bad_rr, "rRInterval"] = np.nan

    # 3. Convert acc/gyr to torch and downsample to ~5 Hz
    acc_20 = torch.tensor(
        df_acc_day[ACC_COLS_RAW].to_numpy(dtype=np.float32).T,
        dtype=torch.float32,
    )  # (3,T20)

    gyr_20 = torch.tensor(
        df_gyr_day[GYR_COLS_RAW].to_numpy(dtype=np.float32).T,
        dtype=torch.float32,
    )  # (3,T20)

    acc_5 = _downsample_20hz_to_5hz(acc_20)  # (3,T5a)
    gyr_5 = _downsample_20hz_to_5hz(gyr_20)  # (3,T5g)

    # hrm is already ~5 Hz
    hr_arr = df_hrm_day["heartRate"].to_numpy(dtype=np.float32)    # (T5h,)
    rr_arr = df_hrm_day["rRInterval"].to_numpy(dtype=np.float32)   # (T5h,)
    mins_arr = _get_minutes_from_timestr_col(df_hrm_day["time"])   # (T5h,)

    # 4. Align all streams to same length
    T5 = min(
        acc_5.shape[1],
        gyr_5.shape[1],
        hr_arr.shape[0],
        rr_arr.shape[0],
        mins_arr.shape[0],
    )
    if T5 <= 0:
        return None, None, None

    acc_5 = acc_5[:, :T5]  # (3,T5)
    gyr_5 = gyr_5[:, :T5]  # (3,T5)
    hr_5 = hr_arr[:T5]     # (T5,)
    rr_5 = rr_arr[:T5]     # (T5,)
    mins = mins_arr[:T5]   # (T5,)

    # 5. Time-of-day sin/cos
    ang = mins * (2.0 * np.pi / (60.0 * 24.0))
    sin_t = np.sin(ang).astype(np.float32)  # (T5,)
    cos_t = np.cos(ang).astype(np.float32)  # (T5,)

    # 6. Stack into channel-major array
    #    acc(3), gyr(3), heart(1), rr(1), sin(1), cos(1)
    day_tensor = np.vstack([
        acc_5.cpu().numpy().astype(np.float32),
        gyr_5.cpu().numpy().astype(np.float32),
        hr_5[None, :].astype(np.float32),
        rr_5[None, :].astype(np.float32),
        sin_t[None, :],
        cos_t[None, :],
    ])  # (10,T5)

    return day_tensor, hr_5, rr_5


def _window_indices_train(T: int, win_len: int, stride_len: int) -> List[Tuple[int, int]]:
    """
    Sliding windows for TRAIN mode using given stride_len.
    """
    if T < win_len:
        return []
    out = []
    for start in range(0, T - win_len, stride_len):
        out.append((start, start + win_len))
    return out


def _window_indices_eval(T: int, win_len: int) -> List[Tuple[int, int]]:
    """
    Less-overlapping windows for VAL/TEST.
    """
    if T < win_len:
        return []
    if T == win_len:
        return [(0, win_len)]
    step = max(1, win_len // 3)
    out = []
    for start in range(0, T - win_len, step):
        out.append((start, start + win_len))
    return out


# -----------------------------
# Dataset
# -----------------------------

class PatientDataset(Dataset):
    """
    Raw dataset for exactly one mode: "train", "val", or "test",
    and (optionally) one patient.

    - Only uses dataset_path.
    - We assume structure like:
        dataset_path/
          P1/
            train_0/
              linacc.parquet
              gyr.parquet
              hrm.parquet
              relapses.csv
            train_1/
              ...
            val_0/
              ...
            test_0/
              ...
          P2/
            ...

    Behavior:
      - TRAIN mode:
          returns overlapping windows:
              data:   (C, T_window)
              target: (5, 1)
              relapse_label = 0 (assumed normal)
      - VAL / TEST mode:
          returns one item per day_index (full-day),
          and __getitem__ slices that into multiple windows for eval:
              data:   (M, C, T_window)
              target: (M, 5)
              relapse_label = 0/1 for that day.
    """

    def __init__(
        self,
        dataset_path: str,
        patient: Optional[str] = None,  # "P1", "P2", ... or None for all
        mode: str = "train",
        scaler: Optional[MinMaxScaler] = None,
        window_size: int = 48,   # in 5-minute bins
        stride: int = 12,        # in 5-minute bins
    ):
        super().__init__()

        self.dataset_path = dataset_path
        self.mode = mode
        self.window_size = window_size
        self.stride = stride

        # Convert "bins" -> raw samples at 5 Hz
        self.window_len_samples = window_size * SAMPLES_PER_BIN    # e.g. 24 * 1500 = 36000
        self.stride_len_samples = stride * SAMPLES_PER_BIN         # e.g. 12 * 1500 = 18000

        # We'll collect:
        # TRAIN: list of window samples directly
        # VAL/TEST: list of whole-day samples
        self.samples = []

        # We'll accumulate (T,10) matrices from TRAIN windows to fit scaler if scaler is None
        scaler_fit_stack = []

        # figure out which patients to iterate
        patients = [patient] if patient is not None else sorted(os.listdir(self.dataset_path))

        for p in patients:
            if p == ".DS_Store":
                continue
            pdir = os.path.join(self.dataset_path, p)
            if not os.path.isdir(pdir):
                continue

            # loop through split folders inside this patient (e.g. train_0, val_0, test_0)
            for subfolder in os.listdir(pdir):
                sub_l = subfolder.lower()

                # match based on prefix:
                is_train_folder = self.mode == "train" and sub_l.startswith("train")
                is_val_folder   = self.mode == "val"   and sub_l.startswith("val")
                is_test_folder  = self.mode == "test"  and sub_l.startswith("test")

                if not (is_train_folder or is_val_folder or is_test_folder):
                    continue

                sdir = os.path.join(pdir, subfolder)

                acc_path = os.path.join(sdir, "linacc.parquet")
                gyr_path = os.path.join(sdir, "gyr.parquet")
                hrm_path = os.path.join(sdir, "hrm.parquet")

                if not (os.path.exists(acc_path) and os.path.exists(gyr_path) and os.path.exists(hrm_path)):
                    continue

                df_acc_all = pd.read_parquet(acc_path)
                df_gyr_all = pd.read_parquet(gyr_path)
                df_hrm_all = pd.read_parquet(hrm_path)

                # relapse labels for that block
                relapse_df = None
                rel_path = os.path.join(sdir, "relapses.csv")
                if os.path.exists(rel_path):
                    relapse_df = pd.read_csv(rel_path)
                    # expected columns: ["day_index","relapse"]

                # list of day_index values in this block
                day_indices = sorted(
                    set(df_acc_all["day_index"].unique())
                    | set(df_gyr_all["day_index"].unique())
                    | set(df_hrm_all["day_index"].unique())
                )

                for day_idx in day_indices:
                    df_acc_day = df_acc_all[df_acc_all["day_index"] == day_idx].copy()
                    df_gyr_day = df_gyr_all[df_gyr_all["day_index"] == day_idx].copy()
                    df_hrm_day = df_hrm_all[df_hrm_all["day_index"] == day_idx].copy()

                    if len(df_acc_day) == 0 or len(df_gyr_day) == 0 or len(df_hrm_day) == 0:
                        continue

                    built = _build_day_tensor(df_acc_day, df_gyr_day, df_hrm_day)
                    if built[0] is None:
                        continue

                    day_tensor, hr_sig, rr_sig = built  # (10,T5), (T5,), (T5,)
                    # impute NaNs per channel so model never sees NaN
                    day_tensor = _impute_nans_channelwise(day_tensor)

                    # fill NaNs in hr/rr just for target calc
                    hr_sig = np.nan_to_num(
                        hr_sig,
                        nan=(np.nanmean(hr_sig) if not np.isnan(np.nanmean(hr_sig)) else 0.0),
                    ).astype(np.float32)
                    rr_sig = np.nan_to_num(
                        rr_sig,
                        nan=(np.nanmean(rr_sig) if not np.isnan(np.nanmean(rr_sig)) else 0.0),
                    ).astype(np.float32)

                    T5 = day_tensor.shape[1]
                    pid_int = int(p[1:]) - 1  # "P3" -> 2

                    # default relapse_label 0
                    relapse_label = 0
                    if relapse_df is not None:
                        try:
                            relapse_label = int(
                                relapse_df[relapse_df["day_index"] == day_idx]["relapse"].values[0]
                            )
                        except Exception:
                            relapse_label = 0

                    if self.mode == "train":
                        # make overlapping windows for training
                        win_idxs = _window_indices_train(
                            T=T5,
                            win_len=self.window_len_samples,
                            stride_len=self.stride_len_samples,
                        )

                        for (s, e) in win_idxs:
                            seg = day_tensor[:, s:e]  # (10, win_len)
                            if seg.shape[1] != self.window_len_samples:
                                continue

                            hr_win = hr_sig[s:e]
                            rr_win = rr_sig[s:e]
                            target_vec = _compute_window_targets(hr_win, rr_win, fs=5.0)  # (5,)

                            self.samples.append({
                                "type": "window",
                                "seq": seg.astype(np.float32),           # (10, win_len)
                                "target": target_vec.astype(np.float32), # (5,)
                                "pid": pid_int,
                                "relapse": 0,  # train is assumed normal
                            })

                            # stash for scaler fit (#timepoints, channels=10)
                            scaler_fit_stack.append(seg.T)

                    else:
                        # val/test: store the entire day
                        self.samples.append({
                            "type": "day",
                            "full_day": day_tensor.astype(np.float32),  # (10,T5)
                            "pid": pid_int,
                            "relapse": relapse_label,
                        })

        # after gathering all samples -> fit / set scaler
        if scaler is None:
            if self.mode == "train" and len(scaler_fit_stack) > 0:
                fit_mat = np.vstack(scaler_fit_stack)  # shape (total_timepoints, 10)
                fit_mat = np.nan_to_num(fit_mat, nan=0.0)
                self.scaler = MinMaxScaler()
                self.scaler.fit(fit_mat)
            else:
                # make a dummy scaler to avoid crash in val/test if no train scaler given
                self.scaler = MinMaxScaler()
                dummy = np.zeros((1, len(ALL_CHANNELS)), dtype=np.float32)
                self.scaler.fit(dummy)
        else:
            self.scaler = scaler

        print(f"Created RAW dataset `{mode}` for Patient {patient} of size: {len(self)}")

    def __len__(self):
        return len(self.samples)

    def _scale_sequence(self, seq_np: np.ndarray) -> np.ndarray:
        """
        seq_np: (10, T)
        -> apply per-channel MinMax scaling learned from train scaler
        returns scaled (10, T)
        """
        flat = seq_np.T  # (T,10)
        flat = np.nan_to_num(flat, nan=0.0)
        flat_scaled = self.scaler.transform(flat)  # (T,10)
        seq_scaled = flat_scaled.T.astype(np.float32)  # (10,T)
        return seq_scaled

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        if sample["type"] == "window":
            # TRAIN item
            seg = sample["seq"]         # (10, win_len)
            target_vec = sample["target"]  # (5,)
            pid = sample["pid"]
            relapse_label = sample["relapse"]

            seg_scaled = self._scale_sequence(seg)  # (10, win_len)

            sequence_tensor = torch.tensor(seg_scaled, dtype=torch.float32)              # (10,T)
            target_tensor   = torch.tensor(target_vec, dtype=torch.float32).unsqueeze(1) # (5,1)

            return {
                "data": sequence_tensor,  # (C,T)
                "target": target_tensor,  # (5,1)
                "user_id": torch.tensor(pid, dtype=torch.long),
                "relapse_label": torch.tensor(relapse_label, dtype=torch.long),
                "idx": idx,
            }

        # VAL / TEST item
        full_day = sample["full_day"]  # (10,T5)
        pid = sample["pid"]
        relapse_label = sample["relapse"]

        T5 = full_day.shape[1]
        if T5 < self.window_len_samples:
            return None

        win_idxs = _window_indices_eval(
            T=T5,
            win_len=self.window_len_samples,
        )
        if len(win_idxs) == 0:
            return None

        sequences = []
        targets = []

        heart_ch = 6  # heartRate channel index in stacked tensor
        rr_ch = 7     # rRInterval channel index

        for (s, e) in win_idxs:
            seg = full_day[:, s:e]  # (10, win_len)
            if seg.shape[1] != self.window_len_samples:
                continue

            # scale seg channel-wise
            seg_scaled = self._scale_sequence(seg)  # (10, win_len)
            sequences.append(seg_scaled.astype(np.float32))

            # compute window targets
            hr_win = seg[heart_ch, :].astype(np.float32)
            rr_win = seg[rr_ch, :].astype(np.float32)
            target_vec = _compute_window_targets(hr_win, rr_win, fs=5.0)  # (5,)
            targets.append(target_vec.astype(np.float32))

        if len(sequences) == 0:
            return None

        sequences = np.stack(sequences, axis=0)  # (M,10,win_len)
        targets   = np.stack(targets,   axis=0)  # (M,5)

        sequence_tensor = torch.tensor(sequences, dtype=torch.float32)  # (M,10,T)
        target_tensor   = torch.tensor(targets,   dtype=torch.float32)  # (M,5)

        return {
            "data": sequence_tensor,
            "target": target_tensor,
            "user_id": torch.tensor(pid, dtype=torch.long),
            "relapse_label": torch.tensor(relapse_label, dtype=torch.long),
            "idx": idx,
        }
