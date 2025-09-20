import os
import datetime
import pandas as pd
import numpy as np
import pyhrv
import scipy
import argparse
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

valid_ranges = {
    "acc_X": (-19.6, 19.6),
    "acc_Y": (-19.6, 19.6),
    "acc_Z": (-19.6, 19.6),
    "gyr_X": (-573, 573),
    "gyr_Y": (-573, 573),
    "gyr_Z": (-573, 573),
    "heartRate": (0, 255),
    "rRInterval": (0, 2000),
}

def rmssd(x):
    x = x.dropna()
    try:
        rmssd = pyhrv.time_domain.rmssd(x)[0]
    except (ZeroDivisionError, ValueError):
        rmssd = np.nan
    return rmssd

def sdnn(x):
    x = x.dropna()
    try:
        sdnn = pyhrv.time_domain.sdnn(x)[0]
    except (ZeroDivisionError, ValueError):
        sdnn = np.nan
    return sdnn

def lombscargle_power_high(nni):
    l = 0.15 * np.pi / 2
    h = 0.4 * np.pi / 2
    freqs = np.linspace(l, h, 1000)
    hf_lsp = scipy.signal.lombscargle(nni.to_numpy(), nni.index.to_numpy(), freqs, normalize=True)
    return np.trapz(hf_lsp, freqs)

def get_norm(df):
    df = df.dropna()
    return np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2).mean()

def time_encoding(slice):
    mean_timestamp = slice['timecol'].astype('datetime64').mean()
    h = mean_timestamp.hour
    m = mean_timestamp.minute
    time_value = h * 60 + m
    sin_t = np.sin(time_value * (2.0 * np.pi / (60 * 24)))
    cos_t = np.cos(time_value * (2.0 * np.pi / (60 * 24)))
    return sin_t, cos_t

def extract_user_features(patient, phase, dataset_path, features_path):
    """Original feature extraction"""
    print(f'Extracting features for patient {patient} phase {phase}')

    # Linear acceleration
    linacc_file = f'{dataset_path}/{patient}/{phase}/linacc.parquet'
    df_linacc = None
    if os.path.exists(linacc_file):
        df_linacc = pd.read_parquet(linacc_file)
        df_linacc['DateTime'] = df_linacc['time'].apply(lambda t: datetime.datetime.combine(datetime.datetime.today(), t))
        for axis in ['X','Y','Z']:
            df_linacc.loc[(df_linacc[axis] < valid_ranges[f'acc_{axis}'][0]) | 
                          (df_linacc[axis] >= valid_ranges[f'acc_{axis}'][1]), axis] = np.nan
        df_linacc = df_linacc.groupby([df_linacc['day_index'], pd.Grouper(key='DateTime', freq='5Min')]).apply(get_norm)

    # Heart rate
    hrm_file = f'{dataset_path}/{patient}/{phase}/hrm.parquet'
    df_hrm = None
    if os.path.exists(hrm_file):
        df_hrm = pd.read_parquet(hrm_file)
        df_hrm['DateTime'] = df_hrm['time'].apply(lambda t: datetime.datetime.combine(datetime.datetime.today(), t))
        df_hrm.loc[df_hrm['heartRate'] <= valid_ranges['heartRate'][0], 'heartRate'] = np.nan
        df_hrm.loc[df_hrm['heartRate'] > valid_ranges['heartRate'][1], 'heartRate'] = np.nan
        df_hrm.loc[df_hrm['rRInterval'] <= valid_ranges['rRInterval'][0], 'rRInterval'] = np.nan
        df_hrm.loc[df_hrm['rRInterval'] > valid_ranges['rRInterval'][1], 'rRInterval'] = np.nan
        df_hrm = df_hrm.groupby([df_hrm['day_index'], pd.Grouper(key='DateTime', freq='5Min')]).agg({
            'heartRate': np.nanmean,
            'rRInterval': [np.nanmean, rmssd, sdnn, lombscargle_power_high]
        })

    # Combine
    df = pd.concat([df_linacc, df_hrm], axis=1).reset_index()
    if 'DateTime' in df.columns:
        h = df['DateTime'].dt.hour
        m = df['DateTime'].dt.minute
        time_value = h * 60 + m
        df['sin_t'] = np.sin(time_value * (2.0 * np.pi / (60 * 24)))
        df['cos_t'] = np.cos(time_value * (2.0 * np.pi / (60 * 24)))
        df = df.drop(columns=['DateTime'])

    new_columns = ['day_index','acc_norm','heartRate_mean','rRInterval_mean','rRInterval_rmssd',
                   'rRInterval_sdnn','rRInterval_lombscargle_power_high','sin_t','cos_t']
    df.columns = new_columns

    # Save
    os.makedirs(f'{features_path}/{patient}/{phase}', exist_ok=True)
    df.to_csv(f'{features_path}/{patient}/{phase}/features.csv', index=False)
    print(f'Saved features for patient {patient} phase {phase}')
    return df

def compute_patient_clusters(dataset_path, n_clusters=3):
    """Compute clusters based on steps + sleep features"""
    patient_stats = []
    patients = sorted(os.listdir(dataset_path))
    for patient in patients:
        steps_file = os.path.join(dataset_path, patient, 'steps.parquet')
        sleep_file = os.path.join(dataset_path, patient, 'sleep.parquet')

        if not os.path.exists(steps_file) or not os.path.exists(sleep_file):
            continue

        steps_df = pd.read_parquet(steps_file)
        sleep_df = pd.read_parquet(sleep_file)

        if len(steps_df)==0 or len(sleep_df)==0:
            continue

        # Simple stats
        total_steps = steps_df['totalSteps'].sum()
        avg_steps = steps_df['totalSteps'].mean()
        sleep_hours = ((pd.to_datetime(sleep_df['end_time']) - pd.to_datetime(sleep_df['start_time'])).dt.total_seconds() / 3600).mean()
        patient_stats.append([patient, total_steps, avg_steps, sleep_hours])

    if len(patient_stats) == 0:
        print("No patient has steps+sleep data for clustering.")
        return {}

    patient_stats_df = pd.DataFrame(patient_stats, columns=['patient','total_steps','avg_steps','sleep_hours'])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(patient_stats_df.iloc[:,1:])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X_scaled)
    patient_stats_df['cluster_id'] = kmeans.labels_

    # Plot clusters
    plt.figure(figsize=(8,6))
    for c in range(n_clusters):
        cluster_df = patient_stats_df[patient_stats_df['cluster_id']==c]
        plt.scatter(cluster_df['avg_steps'], cluster_df['sleep_hours'], label=f'Cluster {c}', s=100)
    plt.xlabel('Average Steps')
    plt.ylabel('Average Sleep Hours')
    plt.title('Patient Behavioral Clusters')
    plt.legend()
    plt.show()

    print(patient_stats_df[['patient','cluster_id']])
    return dict(zip(patient_stats_df['patient'], patient_stats_df['cluster_id']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='path to raw downloaded data')
    parser.add_argument('--out_features_path', type=str, required=True)
    parser.add_argument('--n_jobs', type=int, default=8)
    parser.add_argument('--n_clusters', type=int, default=3)
    args = parser.parse_args()

    patients = os.listdir(args.dataset_path)
    combs = []
    for patient in patients:
        for phase in os.listdir(os.path.join(args.dataset_path,patient)):
            combs.append([patient, phase])

    from joblib import Parallel, delayed
    Parallel(n_jobs=args.n_jobs)(
        delayed(extract_user_features)(patient, phase, args.dataset_path, args.out_features_path) for patient, phase in combs
    )

    # Compute clusters after feature extraction
    patient_clusters = compute_patient_clusters(args.dataset_path, n_clusters=args.n_clusters)
