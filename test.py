from pprint import pprint
import torch
import argparse
import pickle
import pandas as pd
import numpy as np
import os
import sklearn.metrics
from model import TransformerDualHead
from trainer import create_ensemble_mlp


# -------------------
# Helper: cyclical encoding for time
# -------------------
def calculate_sincos_from_minutes(minutes):
    time_value = minutes * (2.0 * np.pi / (60 * 24))
    sin_t = np.sin(time_value)
    cos_t = np.cos(time_value)
    return sin_t, cos_t


# -------------------
# Argument parser
# -------------------
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=24)
    parser.add_argument("--input_features", type=int, default=8)
    parser.add_argument("--output_dim", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--nlayers", type=int, default=2)
    parser.add_argument("--ensembles", type=int, default=5)
    parser.add_argument("--num_patients", type=int, default=8)

    # paths
    parser.add_argument("--features_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--submission_path", type=str, default="/var/tmp/spgc-submission")
    parser.add_argument("--load_path", type=str, required=True)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mode", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()
    args.seq_len = args.window_size
    return args


# -------------------
# Main
# -------------------
def main():
    args = parse()
    pprint(vars(args))
    device = args.device
    print("Using device:", device)

    # --- columns used for scaling ---
    columns_to_scale = [
        "acc_norm",
        "gyr_norm",
        "heartRate_mean",
        "rRInterval_mean",
        "rRInterval_rmssd",
        "rRInterval_sdnn",
        "rRInterval_lombscargle_power_high",
        "steps",
    ]

    # --- Load trained models, ensembles, scalers, and training anomaly distributions ---
    encoders, mlps, scalers, train_dists = [], [], [], []

    for i in range(args.num_patients):
        pth = os.path.join(args.load_path, str(i + 1))

        # load encoder
        encoder = TransformerDualHead(vars(args)).to(device)
        state = torch.load(os.path.join(pth, "best_encoder.pth"), map_location="cpu")
        encoder.load_state_dict(state)
        encoder.eval()
        encoders.append(encoder)

        # load ensemble head
        ensemble = create_ensemble_mlp(args)
        state = torch.load(os.path.join(pth, "best_ensembles.pth"), map_location="cpu")
        ensemble.load_state_dict(state)
        ensemble.eval()
        mlps.append(ensemble)

        # load scaler
        with open(os.path.join(pth, "scaler.pkl"), "rb") as f:
            scalers.append(pickle.load(f))

        # load training anomaly score distribution
        with open(os.path.join(pth, "train_dist_anomaly_scores.pkl"), "rb") as f:
            train_dist_dict = pickle.load(f)
            train_dists.append(train_dist_dict[i])

    torch.set_grad_enabled(False)

    all_auroc, all_auprc = [], []

    # --- Evaluate per patient ---
    for patient in sorted(os.listdir(args.features_path)):
        if patient == ".DS_Store":
            continue

        patient_id = int(patient[1:]) - 1
        if patient_id >= len(encoders):
            continue

        encoder = encoders[patient_id]
        ensemble = mlps[patient_id]
        scaler = scalers[patient_id]
        train_dist = train_dists[patient_id]

        patient_dir = os.path.join(args.features_path, patient)
        user_preds, user_labels = [], []

        for subfolder in os.listdir(patient_dir):
            if (args.mode == "val" and "val" in subfolder and subfolder.endswith("val")) or (
                args.mode == "test" and "test" in subfolder
            ):

                subdir = os.path.join(patient_dir, subfolder)
                feature_path = os.path.join(subdir, "features_stretched_w_steps.csv")
                if not os.path.exists(feature_path):
                    continue

                df = pd.read_csv(feature_path).replace([np.inf, -np.inf], np.nan).dropna()

                # ensure sin_t / cos_t columns
                if not set(["sin_t", "cos_t"]).issubset(df.columns):
                    sin_t, cos_t = calculate_sincos_from_minutes(df["mins"])
                    df["sin_t"] = sin_t
                    df["cos_t"] = cos_t

                # relapse labels
                if args.mode == "test":
                    relapse_path = os.path.join(args.dataset_path, patient, subfolder, "relapses.csv")
                    relapse_df = pd.read_csv(relapse_path)
                    relapse_df = relapse_df.iloc[:-1]
                    DAY_INDEX = "day_index"
                else:
                    relapse_path = os.path.join(subdir, "relapse_stretched.csv")
                    relapse_df = pd.read_csv(relapse_path)
                    DAY_INDEX = "day"

                for day_index in relapse_df[DAY_INDEX].unique():
                    day_data = df[df[DAY_INDEX] == day_index]
                    if len(day_data) < args.window_size:
                        anomaly_score = 0.0
                    else:
                        # create sliding windows
                        sequences = []
                        if len(day_data) == args.window_size:
                            seq = day_data.iloc[0:args.window_size][columns_to_scale].to_numpy()
                            sequences.append(scaler.transform(seq))
                        else:
                            for start in range(0, len(day_data) - args.window_size, args.window_size // 3):
                                seq = day_data.iloc[start:start + args.window_size][columns_to_scale].to_numpy()
                                sequences.append(scaler.transform(seq))

                        sequence = np.stack(sequences)
                        seq_tensor = torch.tensor(sequence, dtype=torch.float32).permute(0, 2, 1).to(device)

                        # forward through encoder
                        features, preds_time, preds_sleep = encoder(seq_tensor)

                        # combine both circadian and sleep variance signals
                        k = args.ensembles
                        batched_features = features[None, :, :].repeat([k, 1, 1])
                        preds = ensemble(batched_features)
                        avg_pred = torch.mean(preds, 0)
                        var_time = torch.sum((preds - avg_pred) ** 2, dim=(2,))
                        mean_var_time = torch.mean(torch.mean(var_time, 0)).item()

                        # also compute variance in sleep predictions for added sensitivity
                        sleep_mean = torch.mean(preds_sleep, 0)
                        sleep_var = torch.mean((preds_sleep - sleep_mean) ** 2).item()

                        # combine time & sleep anomaly measures
                        combined_var = 0.7 * mean_var_time + 0.3 * sleep_var

                        _mean = np.mean(train_dist)
                        _max, _min = np.max(train_dist), np.min(train_dist)
                        anomaly_score = (combined_var - _mean) / (_max - _min)
                        anomaly_score = 1.0 if anomaly_score > 0 else 0.0

                    relapse_df.loc[relapse_df[DAY_INDEX] == day_index, "score"] = anomaly_score
                    user_preds.append(anomaly_score)

                    # collect labels if available
                    if "relapse" in relapse_df.columns:
                        relapse_label = relapse_df[relapse_df[DAY_INDEX] == day_index]["relapse"].to_numpy()[0]
                        user_labels.append(relapse_label)

                # save predictions
                if args.mode == "test":
                    save_dir = os.path.join(args.submission_path, f"patient{patient[1]}", subfolder)
                    os.makedirs(save_dir, exist_ok=True)
                    relapse_df.to_csv(os.path.join(save_dir, "submission.csv"), index=False)
                    print(f"Saved submission to: {save_dir}")

        # --- Evaluation: now prints metrics in both test and val ---
        if len(np.unique(user_labels)) > 1:
            y_true, y_pred = np.array(user_labels), np.array(user_preds)
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)
            precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_pred)
            auroc = sklearn.metrics.auc(fpr, tpr)
            auprc = sklearn.metrics.auc(recall, precision)
            all_auroc.append(auroc)
            all_auprc.append(auprc)
            print(f"USER {patient}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, AVG={(auroc + auprc) / 2:.4f}")
        else:
            print(f"USER {patient}: skipped metrics (labels missing or constant).")

    # --- Final Aggregate Metrics (printed in both test & val) ---
    if all_auroc and all_auprc:
        total_auroc = np.mean(all_auroc)
        total_auprc = np.mean(all_auprc)
        total_avg = (total_auroc + total_auprc) / 2.0
        print(f"TOTAL AUROC={total_auroc:.4f}, TOTAL AUPRC={total_auprc:.4f}, AVG={total_avg:.4f}")


if __name__ == "__main__":
    main()
