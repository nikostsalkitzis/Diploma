import torch
import argparse
import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
from pprint import pprint
import sklearn.metrics

from model import CNNLSTMClassifier
from trainer import create_ensemble_mlp
from dataset import calculate_sincos_from_minutes


# ------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------
def parse():
    parser = argparse.ArgumentParser()

    # Model / ensemble parameters
    parser.add_argument('--window_size', type=int, default=24)
    parser.add_argument('--stride', type=int, default=12)
    parser.add_argument('--input_features', type=int, default=8)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--ensembles', type=int, default=5)
    parser.add_argument('--num_patients', type=int, default=8)

    # Data paths
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--submission_path', type=str, default='submissions')
    parser.add_argument('--load_path', type=str, default='checkpoints')

    # Mode: val or test
    parser.add_argument('--mode', type=str, default='test', choices=['val', 'test'])

    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    args.seq_len = args.window_size
    return args


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    args = parse()
    pprint(vars(args))

    device = args.device
    print(f"\nUsing device: {device}")

    # Columns used for scaling and input
    columns_to_scale = [
        'acc_norm', 'gyr_norm', 'heartRate_mean', 'rRInterval_mean',
        'rRInterval_rmssd', 'rRInterval_sdnn', 'rRInterval_lombscargle_power_high',
        'steps'
    ]

    # ------------------------------------------------------------
    # 1️⃣ Load models, MLPs, and scalers
    # ------------------------------------------------------------
    encoders = [CNNLSTMClassifier(vars(args)).to(device) for _ in range(args.num_patients)]
    mlps = [create_ensemble_mlp(args) for _ in range(args.num_patients)]

    scalers, train_dists = [], []

    print("\nLoading checkpoints...")
    for i in range(args.num_patients):
        patient_dir = os.path.join(args.load_path, str(i + 1))
        if not os.path.exists(patient_dir):
            print(f"⚠️ Skipping missing checkpoint for patient {i + 1}")
            scalers.append(None)
            train_dists.append(None)
            continue

        # Load encoder and ensemble
        encoders[i].load_state_dict(torch.load(os.path.join(patient_dir, 'best_encoder.pth'), map_location='cpu'))
        mlps[i].load_state_dict(torch.load(os.path.join(patient_dir, 'best_ensembles.pth'), map_location='cpu'))
        encoders[i].eval()
        mlps[i].eval()

        # Load scaler
        with open(os.path.join(patient_dir, 'scaler.pkl'), 'rb') as f:
            scalers.append(pickle.load(f))
        # Load training variance distribution
        with open(os.path.join(patient_dir, 'train_dist_anomaly_scores.pkl'), 'rb') as f:
            train_dists.append(pickle.load(f))

    torch.set_grad_enabled(False)

    # ------------------------------------------------------------
    # 2️⃣ Process each patient
    # ------------------------------------------------------------
    all_auroc, all_auprc = [], []

    for patient in sorted(os.listdir(args.features_path)):
        if patient == ".DS_Store":
            continue
        patient_id = int(patient[1:]) - 1
        if patient_id >= len(encoders):
            continue

        encoder = encoders[patient_id]
        mlp = mlps[patient_id]
        scaler = scalers[patient_id]
        train_dist = train_dists[patient_id]

        if scaler is None or train_dist is None:
            print(f"⚠️ Missing data for {patient}, skipping.")
            continue

        print(f"\nProcessing {patient}...")

        patient_dir = os.path.join(args.features_path, patient)
        user_anomaly_scores, user_relapse_labels = [], []

        for subfolder in os.listdir(patient_dir):
            if (args.mode == 'val' and 'val' in subfolder and subfolder.endswith('val')) or \
               (args.mode == 'test' and 'test' in subfolder):

                subfolder_dir = os.path.join(patient_dir, subfolder)
                feature_path = os.path.join(subfolder_dir, 'features_stretched_w_steps.csv')

                if not os.path.exists(feature_path):
                    continue

                # Skip already processed (useful if rerunning)
                if args.mode == 'test':
                    save_dir = os.path.join(args.submission_path, f'patient{patient[1]}', subfolder)
                    save_file = os.path.join(save_dir, 'submission.csv')
                    if os.path.exists(save_file):
                        print(f"Skipping already processed {patient}/{subfolder}")
                        continue

                df = pd.read_csv(feature_path)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()

                # ensure sin_t, cos_t exist
                if 'sin_t' not in df.columns or 'cos_t' not in df.columns:
                    mins = df['mins']
                    sin_t, cos_t = calculate_sincos_from_minutes(mins)
                    df['sin_t'], df['cos_t'] = sin_t, cos_t

                # load relapse info
                if args.mode == 'test':
                    relapse_path = os.path.join(args.dataset_path, patient, subfolder, 'relapses.csv')
                    relapse_df = pd.read_csv(relapse_path)
                    relapse_df = relapse_df.iloc[:-1]  # drop last invalid row if needed
                    DAY_INDEX = 'day_index'
                else:
                    relapse_path = os.path.join(args.features_path, patient, subfolder, 'relapse_stretched.csv')
                    relapse_df = pd.read_csv(relapse_path)
                    DAY_INDEX = 'day'

                day_indices = relapse_df[DAY_INDEX].unique()

                for day_index in tqdm(day_indices, desc=f"{patient}/{subfolder}"):
                    day_data = df[df[DAY_INDEX] == day_index].copy()
                    if len(day_data) < args.window_size:
                        user_anomaly_scores.append(0.0)
                        if args.mode != 'test':
                            relapse_label = relapse_df[relapse_df[DAY_INDEX] == day_index]['relapse'].values[0]
                            user_relapse_labels.append(relapse_label)
                        continue

                    # prepare overlapping sequences (optimized)
                    sequences = []
                    if len(day_data) == args.window_size:
                        seq = day_data.iloc[:args.window_size]
                        seq = scaler.transform(seq[columns_to_scale].to_numpy())
                        sequences.append(seq)
                    else:
                        # less overlap => faster
                        for start_idx in range(0, len(day_data) - args.window_size, args.window_size):
                            seq = day_data.iloc[start_idx:start_idx + args.window_size]
                            seq = scaler.transform(seq[columns_to_scale].to_numpy())
                            sequences.append(seq)

                    sequence = np.stack(sequences)
                    seq_tensor = torch.tensor(sequence, dtype=torch.float32).permute(0, 2, 1).to(device)

                    # forward pass
                    features, _ = encoder(seq_tensor)
                    batched_features = features[None, :, :].repeat([args.ensembles, 1, 1])
                    preds = mlp(batched_features)
                    avg_pred = torch.mean(preds, 0)

                    # compute variance-based anomaly score
                    _mean = np.mean(train_dist[patient_id])
                    _max = np.max(train_dist[patient_id])
                    _min = np.min(train_dist[patient_id])
                    var_score = torch.sum((preds - avg_pred) ** 2, dim=(2,))
                    mean_var = torch.mean(torch.mean(var_score, 0)).item()
                    anomaly_score = (mean_var - _mean) / (_max - _min)
                    anomaly_score = float(anomaly_score > 0.0)
                    user_anomaly_scores.append(anomaly_score)

                    if args.mode != 'test':
                        relapse_label = relapse_df[relapse_df[DAY_INDEX] == day_index]['relapse'].values[0]
                        user_relapse_labels.append(relapse_label)
                        relapse_df.loc[relapse_df[DAY_INDEX] == day_index, 'score'] = anomaly_score

                # save predictions for test mode
                if args.mode == 'test':
                    os.makedirs(save_dir, exist_ok=True)
                    relapse_df.to_csv(save_file, index=False)
                    print(f"✅ Saved → {save_file}")

        # --------------------------------------------------------
        # Evaluation for validation mode
        # --------------------------------------------------------
        if args.mode != 'test' and len(user_anomaly_scores) > 0:
            user_anomaly_scores = np.array(user_anomaly_scores)
            user_relapse_labels = np.array(user_relapse_labels)

            precision, recall, _ = sklearn.metrics.precision_recall_curve(user_relapse_labels, user_anomaly_scores)
            fpr, tpr, _ = sklearn.metrics.roc_curve(user_relapse_labels, user_anomaly_scores)
            auroc = sklearn.metrics.auc(fpr, tpr)
            auprc = sklearn.metrics.auc(recall, precision)

            all_auroc.append(auroc)
            all_auprc.append(auprc)
            print(f"{patient}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, AVG={(auroc + auprc)/2:.4f}")

    # ------------------------------------------------------------
    # 3️⃣ Global metrics summary
    # ------------------------------------------------------------
    if args.mode != 'test' and len(all_auroc) > 0:
        total_auroc = np.mean(all_auroc)
        total_auprc = np.mean(all_auprc)
        total_avg = (total_auroc + total_auprc) / 2
        print("\n================ RESULTS ================")
        print(f"Total AUROC: {total_auroc:.4f}")
        print(f"Total AUPRC: {total_auprc:.4f}")
        print(f"Total AVG  : {total_avg:.4f}")
        print("=========================================")


if __name__ == "__main__":
    main()
