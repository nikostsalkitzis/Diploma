from pprint import pprint
import torch
import argparse
from model import TransformerClassifier
from dataset import calculate_sincos_from_minutes
from trainer import create_ensemble_mlp
import pickle
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import sklearn.metrics


def parse():
    parser = argparse.ArgumentParser(description="Evaluate multi-task ensemble model for relapse detection")

    # transformer / ensemble settings
    parser.add_argument('--window_size', type=int, default=24)
    parser.add_argument('--input_features', type=int, default=8)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--nlayers', type=int, default=2)
    parser.add_argument('--ensembles', type=int, default=5)
    parser.add_argument('--num_patients', type=int, default=8)

    # paths
    parser.add_argument('--features_path', type=str, required=True, help='Path to feature folders per patient')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to dataset (for relapse labels)')
    parser.add_argument('--load_path', type=str, required=True, help='Root directory of saved models')
    parser.add_argument('--submission_path', type=str, default='submissions', help='Where to save prediction CSVs')

    # mode
    parser.add_argument('--mode', type=str, default='test', choices=['val', 'test'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    args.seq_len = args.window_size
    return args


def flatten_train_dist(train_dist_anomaly_scores):
    """Safely extract numeric values from possibly nested dicts/lists."""
    if isinstance(train_dist_anomaly_scores, dict):
        values = []
        for v in train_dist_anomaly_scores.values():
            if isinstance(v, (list, np.ndarray)):
                values.extend(v)
            else:
                values.append(v)
        return np.array(values)
    return np.array(train_dist_anomaly_scores)


def main():
    args = parse()
    pprint(vars(args))

    device = args.device
    os.makedirs(args.submission_path, exist_ok=True)

    print(f"Running on device: {device}")

    # Initialize models and ensemble heads
    encoders = [TransformerClassifier(vars(args)).to(device) for _ in range(args.num_patients)]
    mlps = [create_ensemble_mlp(args) for _ in range(args.num_patients)]

    scalers, train_dists = [], []

    # ---- Load checkpoints and scalers ----
    for i in range(args.num_patients):
        patient_folder = os.path.join(args.load_path, str(i + 1))
        print(f"Loading model for patient {i + 1} from {patient_folder}")

        # Load encoder weights
        enc_path = os.path.join(patient_folder, 'best_encoder.pth')
        encoders[i].load_state_dict(torch.load(enc_path, map_location=device))
        encoders[i].eval()

        # Load ensemble weights
        ens_path = os.path.join(patient_folder, 'best_ensembles.pth')
        mlps[i].load_state_dict(torch.load(ens_path, map_location=device))
        mlps[i].eval()

        # Load scaler
        with open(os.path.join(patient_folder, 'scaler.pkl'), 'rb') as f:
            scalers.append(pickle.load(f))

        # Load train distribution anomaly scores
        with open(os.path.join(patient_folder, 'train_dist_anomaly_scores.pkl'), 'rb') as f:
            train_dists.append(pickle.load(f))

    torch.set_grad_enabled(False)

    # ---- Evaluate each patient ----
    all_auroc, all_auprc = [], []

    for patient in sorted(os.listdir(args.features_path)):
        if patient.startswith('.'):
            continue
        patient_id = int(patient[1:]) - 1
        if patient_id >= len(encoders):
            continue

        print(f"\nEvaluating Patient {patient}")

        patient_dir = os.path.join(args.features_path, patient)
        encoder, ensemble_head = encoders[patient_id], mlps[patient_id]
        scaler = scalers[patient_id]
        train_dist_anomaly_scores = flatten_train_dist(train_dists[patient_id])

        user_anomaly_scores, user_relapse_labels = [], []

        for subfolder in os.listdir(patient_dir):
            if (args.mode == 'val' and 'val' in subfolder and subfolder.endswith('val')) \
               or (args.mode == 'test' and 'test' in subfolder):

                subfolder_dir = os.path.join(patient_dir, subfolder)
                feat_file = os.path.join(subfolder_dir, 'features_stretched_w_steps.csv')
                if not os.path.exists(feat_file):
                    continue

                df = pd.read_csv(feat_file).replace([np.inf, -np.inf], np.nan).dropna()
                if 'sin_t' not in df.columns or 'cos_t' not in df.columns:
                    mins = df['mins']
                    sin_t, cos_t = calculate_sincos_from_minutes(mins)
                    df['sin_t'], df['cos_t'] = sin_t, cos_t

                # Load relapse labels
                if args.mode == 'test':
                    relapse_path = os.path.join(args.dataset_path, patient, subfolder, 'relapses.csv')
                    relapse_df = pd.read_csv(relapse_path)
                    relapse_df = relapse_df.iloc[:-1]  # drop last dummy row
                else:
                    relapse_path = os.path.join(args.features_path, patient, subfolder, 'relapse_stretched.csv')
                    relapse_df = pd.read_csv(relapse_path)

                day_col = 'day_index' if args.mode == 'test' else 'day'
                day_indices = relapse_df[day_col].unique()

                for day_idx in day_indices:
                    day_data = df[df[day_col] == day_idx]
                    relapse_label = relapse_df.loc[relapse_df[day_col] == day_idx, 'relapse'].values[0] \
                                    if 'relapse' in relapse_df.columns else 0

                    if len(day_data) < args.window_size:
                        user_anomaly_scores.append(0)
                        user_relapse_labels.append(relapse_label)
                        continue

                    # Generate overlapping sequences
                    sequences = []
                    for start in range(0, len(day_data) - args.window_size + 1, args.window_size // 3):
                        seq = day_data.iloc[start:start + args.window_size][[
                            'acc_norm', 'gyr_norm', 'heartRate_mean', 'rRInterval_mean',
                            'rRInterval_rmssd', 'rRInterval_sdnn',
                            'rRInterval_lombscargle_power_high', 'steps'
                        ]].to_numpy()
                        seq = scaler.transform(seq)
                        sequences.append(seq)
                    sequences = np.stack(sequences)

                    # Convert to tensor (batch, seq_len, features)
                    seq_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)

                    # ---- Shape sanity check ----
                    if seq_tensor.shape[-1] != args.input_features:
                        raise ValueError(f"[ShapeError] Expected {args.input_features} features, "
                                         f"got {seq_tensor.shape[-1]} at patient {patient}, day {day_idx}")

                    # Forward pass
                    outputs = encoder(seq_tensor)
                    features = outputs['features']  # (batch, d_model)
                    k = args.ensembles
                    batched_features = features[None, :, :].repeat([k, 1, 1])

                    preds = ensemble_head(batched_features)
                    avg_pred = torch.mean(preds, 0)

                    _mean = np.mean(train_dist_anomaly_scores)
                    _max, _min = np.max(train_dist_anomaly_scores), np.min(train_dist_anomaly_scores)

                    var_score = torch.sum((preds - avg_pred) ** 2, dim=2)
                    mean_var = torch.mean(torch.mean(var_score, 0)).item()
                    anomaly_score = (mean_var - _mean) / (_max - _min)
                    anomaly_score = float(anomaly_score > 0.0)

                    relapse_df.loc[relapse_df[day_col] == day_idx, 'score'] = anomaly_score
                    user_anomaly_scores.append(anomaly_score)
                    user_relapse_labels.append(relapse_label)

                # Save per-subfolder predictions
                save_dir = os.path.join(args.submission_path, f'patient{patient[1]}', subfolder)
                os.makedirs(save_dir, exist_ok=True)
                relapse_df.to_csv(os.path.join(save_dir, 'submission.csv'), index=False)
                print(f"Saved predictions → {save_dir}/submission.csv")

        # ---- Compute validation metrics ----
        if args.mode != 'test' and len(user_anomaly_scores) > 0:
            user_anomaly_scores = np.array(user_anomaly_scores)
            user_relapse_labels = np.array(user_relapse_labels)

            precision, recall, _ = sklearn.metrics.precision_recall_curve(user_relapse_labels, user_anomaly_scores)
            fpr, tpr, _ = sklearn.metrics.roc_curve(user_relapse_labels, user_anomaly_scores)
            auroc = sklearn.metrics.auc(fpr, tpr)
            auprc = sklearn.metrics.auc(recall, precision)

            all_auroc.append(auroc)
            all_auprc.append(auprc)
            print(f"Patient {patient}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, AVG={(auroc + auprc)/2:.4f}")

    # ---- Print global metrics ----
    if args.mode != 'test' and len(all_auroc) > 0:
        total_auroc = np.mean(all_auroc)
        total_auprc = np.mean(all_auprc)
        total_avg = (total_auroc + total_auprc) / 2
        print("\n========== Final Validation Results ==========")
        print(f"Total AUROC: {total_auroc:.4f}")
        print(f"Total AUPRC: {total_auprc:.4f}")
        print(f"Total AVG:   {total_avg:.4f}")


if __name__ == '__main__':
    main()
