from pprint import pprint

import torch
import argparse
from model import TransformerClassifier
from trainer import create_ensemble_mlp
import pickle
import pandas as pd
import os
import numpy as np
import sklearn.metrics


def calculate_sincos_from_minutes(minutes):
    time_value = minutes * (2.0 * np.pi / (60 * 24))
    sin_t = np.sin(time_value)
    cos_t = np.cos(time_value)
    return sin_t, cos_t


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


def main():
    args = parse()
    pprint(vars(args))
    device = args.device
    print("Using device:", device)

    # same columns used in training
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

    # load models + ensemble mlps + scalers + train anomaly distributions
    encoders, mlps, scalers, train_dists = [], [], [], []
    for i in range(args.num_patients):
        model = TransformerClassifier(vars(args)).to(device)
        pth = os.path.join(args.load_path, str(i + 1))

        # encoder
        state = torch.load(os.path.join(pth, "best_encoder.pth"), map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        encoders.append(model)

        # ensemble head
        ensemble = create_ensemble_mlp(args)
        state = torch.load(os.path.join(pth, "best_ensembles.pth"), map_location="cpu")
        ensemble.load_state_dict(state)
        ensemble.eval()
        mlps.append(ensemble)

        # scaler
        with open(os.path.join(pth, "scaler.pkl"), "rb") as f:
            scalers.append(pickle.load(f))

        # train anomaly distribution (dict -> select scores for patient i)
        with open(os.path.join(pth, "train_dist_anomaly_scores.pkl"), "rb") as f:
            train_dist_dict = pickle.load(f)
            train_dists.append(train_dist_dict[i])

    torch.set_grad_enabled(False)

    all_auroc, all_auprc = [], []

    for patient in sorted(os.listdir(args.features_path)):
        if patient == ".DS_Store":
            continue

        patient_id = int(patient[1:]) - 1
        if patient_id >= len(encoders):
            continue

        patient_dir = os.path.join(args.features_path, patient)
        user_preds, user_labels = [], []

        for subfolder in os.listdir(patient_dir):
            if (args.mode == "val" and "val" in subfolder and subfolder.endswith("val")) or (
                args.mode == "test" and "test" in subfolder
            ):

                file_path = os.path.join(patient_dir, subfolder, "features_stretched_w_steps.csv")
                df = pd.read_csv(file_path).replace([np.inf, -np.inf], np.nan).dropna()

                if "sin_t" not in df or "cos_t" not in df:
                    sin_t, cos_t = calculate_sincos_from_minutes(df["mins"])
                    df["sin_t"] = sin_t
                    df["cos_t"] = cos_t

                if args.mode == "test":
                    relapse_df = pd.read_csv(os.path.join(args.dataset_path, patient, subfolder, "relapses.csv"))
                    relapse_df = relapse_df.iloc[:-1]
                    DAY_INDEX = "day_index"
                else:
                    relapse_df = pd.read_csv(os.path.join(patient_dir, subfolder, "relapse_stretched.csv"))
                    DAY_INDEX = "day"

                for day_index in relapse_df[DAY_INDEX].unique():
                    day_data = df[df[DAY_INDEX] == day_index]
                    if len(day_data) < args.window_size:
                        anomaly_score = 0.0
                    else:
                        sequences = []
                        if len(day_data) == args.window_size:
                            seq = day_data.iloc[0 : args.window_size][columns_to_scale].to_numpy()
                            sequences.append(scalers[patient_id].transform(seq))
                        else:
                            for start in range(0, len(day_data) - args.window_size, args.window_size // 3):
                                seq = day_data.iloc[start : start + args.window_size][columns_to_scale].to_numpy()
                                sequences.append(scalers[patient_id].transform(seq))
                        sequence = np.stack(sequences)
                        seq_tensor = torch.tensor(sequence, dtype=torch.float32).permute(0, 2, 1).to(device)

                        # forward encoder (take only features)
                        outputs = encoders[patient_id](seq_tensor)
                        features = outputs[0]  # first output = features

                        # forward ensemble
                        k = args.ensembles
                        batched_features = features[None, :, :].repeat([k, 1, 1])
                        preds = mlps[patient_id](batched_features)
                        average_pred = torch.mean(preds, 0)

                        var_score = torch.sum((preds - average_pred) ** 2, dim=(2,))
                        mean_var = torch.mean(torch.mean(var_score, 0)).item()

                        _mean = np.mean(train_dists[patient_id])
                        _max, _min = np.max(train_dists[patient_id]), np.min(train_dists[patient_id])
                        anomaly_score = (mean_var - _mean) / (_max - _min)
                        anomaly_score = 1.0 if anomaly_score > 0 else 0.0

                    relapse_df.loc[relapse_df[DAY_INDEX] == day_index, "score"] = anomaly_score
                    user_preds.append(anomaly_score)
                    if args.mode != "test":
                        relapse_label = relapse_df[relapse_df[DAY_INDEX] == day_index]["relapse"].to_numpy()[0]
                        user_labels.append(relapse_label)

                if args.mode == "test":
                    save_dir = os.path.join(args.submission_path, f"patient{patient[1]}", subfolder)
                    os.makedirs(save_dir, exist_ok=True)
                    relapse_df.to_csv(os.path.join(save_dir, "submission.csv"), index=False)
                    print(f"Saved submission to: {save_dir}")

        if args.mode != "test" and len(np.unique(user_labels)) > 1:
            y_true, y_pred = np.array(user_labels), np.array(user_preds)
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)
            precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_pred)
            auroc = sklearn.metrics.auc(fpr, tpr)
            auprc = sklearn.metrics.auc(recall, precision)
            all_auroc.append(auroc)
            all_auprc.append(auprc)
            print(f"USER {patient}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, AVG={(auroc+auprc)/2:.4f}")

    if args.mode != "test" and all_auroc:
        total_auroc = sum(all_auroc) / len(all_auroc)
        total_auprc = sum(all_auprc) / len(all_auprc)
        print(
            f"TOTAL AUROC={total_auroc:.4f}, TOTAL AUPRC={total_auprc:.4f}, "
            f"AVG={(total_auroc+total_auprc)/2:.4f}"
        )


if __name__ == "__main__":
    main()
