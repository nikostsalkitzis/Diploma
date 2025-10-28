from pprint import pprint
import argparse, os, pickle, numpy as np, pandas as pd, torch, sklearn.metrics
from model import TransformerHeartPredictor
from trainer import create_ensemble_mlp

COLS8 = [
    "acc_norm","gyr_norm","heartRate_mean","rRInterval_mean",
    "rRInterval_rmssd","rRInterval_sdnn","rRInterval_lombscargle_power_high","steps",
]
COLS10 = COLS8 + ["sin_t","cos_t"]

def calculate_sincos_from_minutes(minutes):
    ang = minutes * (2.0 * np.pi / (60 * 24))
    return np.sin(ang), np.cos(ang)

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--window_size", type=int, default=24)
    p.add_argument("--input_features", type=int, default=10)  # default; overridden per ckpt
    p.add_argument("--output_dim", type=int, default=5)
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--ensembles", type=int, default=5)
    p.add_argument("--num_patients", type=int, default=8)
    p.add_argument("--features_path", type=str, required=True)
    p.add_argument("--dataset_path", type=str, required=True)
    p.add_argument("--submission_path", type=str, default="/var/tmp/spgc-submission")
    p.add_argument("--load_path", type=str, required=True)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mode", type=str, default="test", choices=["val","test"])
    args = p.parse_args()
    args.seq_len = args.window_size
    return args

def ensure_time_cols(df):
    if ("sin_t" not in df.columns) or ("cos_t" not in df.columns):
        if "mins" not in df.columns:
            raise ValueError("Missing 'mins' column to compute sin_t/cos_t.")
        sin_t, cos_t = calculate_sincos_from_minutes(df["mins"])
        df = df.copy()
        df["sin_t"] = sin_t
        df["cos_t"] = cos_t
    return df

def map_scaled_to_model(seq_scaled, cols_scaler, cols_model, raw_seq_for_fallback=None):
    idx_map = []
    missing = []
    for c in cols_model:
        if c in cols_scaler:
            idx_map.append(cols_scaler.index(c))
        else:
            missing.append(c)
    out = seq_scaled[:, idx_map] if idx_map else None

    if missing:
        if raw_seq_for_fallback is None:
            raise ValueError(f"Missing columns {missing} in scaler; no fallback provided.")
        extra = np.stack([raw_seq_for_fallback[:, raw_seq_for_fallback_cols.index(c)] for c in missing], axis=1)
        out = np.concatenate([out, extra], axis=1) if out is not None else extra
        reorder = [(cols_model.index(c), i) for i, c in enumerate([x for x in cols_model if x in cols_scaler] + missing)]
        reorder_sorted = [i for _, i in sorted(reorder)]
        out = out[:, reorder_sorted]
    return out

def main():
    args = parse()
    pprint(vars(args))
    device = args.device
    print("Using device:", device)

    encoders, mlps, scalers, train_dists, cols_model_list, cols_scaler_list = [], [], [], [], [], []

    for i in range(args.num_patients):
        pdir = os.path.join(args.load_path, str(i+1))
        enc_pth = os.path.join(pdir, "best_encoder.pth")
        ens_pth = os.path.join(pdir, "best_ensembles.pth")
        scaler_pth = os.path.join(pdir, "scaler.pkl")
        dist_pth = os.path.join(pdir, "train_dist_anomaly_scores.pkl")
        if not (os.path.exists(enc_pth) and os.path.exists(ens_pth) and os.path.exists(scaler_pth) and os.path.exists(dist_pth)):
            print(f"⚠️ Skipping patient {i+1}: missing artifacts in {pdir}")
            encoders.append(None); mlps.append(None); scalers.append(None)
            train_dists.append(None); cols_model_list.append(None); cols_scaler_list.append(None)
            continue

        state = torch.load(enc_pth, map_location="cpu")
        in_feats_ckpt = state["encoder_input_layer.weight"].shape[1]
        cols_model = COLS10 if in_feats_ckpt == 10 else COLS8
        cols_model_list.append(cols_model)

        local_args = vars(args).copy()
        local_args["input_features"] = in_feats_ckpt
        model = TransformerHeartPredictor(local_args).to(device)
        model.load_state_dict(state, strict=True)
        model.eval()
        encoders.append(model)

        ensemble = create_ensemble_mlp(args)
        ensemble.load_state_dict(torch.load(ens_pth, map_location=device), strict=True)
        ensemble.eval()
        mlps.append(ensemble)

        with open(scaler_pth, "rb") as f:
            scaler = pickle.load(f)
        scalers.append(scaler)
        n_in = getattr(scaler, "n_features_in_", None)
        if n_in not in (8,10):
            raise ValueError(f"Scaler for patient {i+1} has unexpected n_features_in_={n_in}")
        cols_scaler = COLS10 if n_in == 10 else COLS8
        cols_scaler_list.append(cols_scaler)

        with open(dist_pth, "rb") as f:
            td = pickle.load(f)
            train_dists.append(td[i])

    torch.set_grad_enabled(False)
    all_auroc, all_auprc = [], []

    for patient in sorted(os.listdir(args.features_path)):
        if patient == ".DS_Store": continue
        pid = int(patient[1:]) - 1
        if pid >= len(encoders) or encoders[pid] is None: continue

        encoder = encoders[pid]; ensemble = mlps[pid]; scaler = scalers[pid]
        train_dist = train_dists[pid]; cols_model = cols_model_list[pid]; cols_scaler = cols_scaler_list[pid]

        pdir = os.path.join(args.features_path, patient)
        user_preds, user_labels = [], []

        for sub in os.listdir(pdir):
            if (args.mode == "val" and "val" in sub and sub.endswith("val")) or (args.mode == "test" and "test" in sub):
                fpath = os.path.join(pdir, sub, "features_stretched_w_steps.csv")
                if not os.path.exists(fpath): continue

                df = pd.read_csv(fpath).replace([np.inf,-np.inf], np.nan).dropna()
                df = ensure_time_cols(df)

                if args.mode == "test":
                    relapse_df = pd.read_csv(os.path.join(args.dataset_path, patient, sub, "relapses.csv"))
                    relapse_df = relapse_df.iloc[:-1]
                    DAY_INDEX = "day_index"
                else:
                    relapse_df = pd.read_csv(os.path.join(pdir, sub, "relapse_stretched.csv"))
                    DAY_INDEX = "day"

                global raw_seq_for_fallback_cols
                raw_seq_for_fallback_cols = COLS10

                for day_idx in relapse_df[DAY_INDEX].unique():
                    day_df = df[df[DAY_INDEX] == day_idx]
                    if len(day_df) < args.window_size:
                        relapse_df.loc[relapse_df[DAY_INDEX] == day_idx, "score"] = 0.0
                        user_preds.append(0.0)
                        if "relapse" in relapse_df.columns:
                            user_labels.append(relapse_df[relapse_df[DAY_INDEX]==day_idx]["relapse"].to_numpy()[0])
                        continue

                    sequences = []
                    step = max(1, args.window_size // 3)
                    starts = [0] if len(day_df) == args.window_size else range(0, len(day_df)-args.window_size, step)
                    for s in starts:
                        seq_scaler = day_df.iloc[s:s+args.window_size][cols_scaler].to_numpy()
                        seq_scaled = scaler.transform(seq_scaler)
                        raw_seq = day_df.iloc[s:s+args.window_size][COLS10].to_numpy()
                        seq_model = map_scaled_to_model(seq_scaled, cols_scaler, cols_model, raw_seq_for_fallback=raw_seq)
                        sequences.append(seq_model)

                    sequence = np.stack(sequences)
                    seq_tensor = torch.tensor(sequence, dtype=torch.float32).permute(0,2,1).to(device)

                    features, _ = encoder(seq_tensor)

                    k = args.ensembles
                    bf = features[None, :, :].repeat([k,1,1])
                    preds = ensemble(bf)
                    avg = torch.mean(preds, 0)
                    var_score = torch.sum((preds - avg) ** 2, dim=2)
                    mean_var = torch.mean(torch.mean(var_score, 0)).item()

                    mu = float(np.mean(train_dist)); mx = float(np.max(train_dist)); mn = float(np.min(train_dist))
                    denom = (mx - mn) if (mx - mn) != 0 else 1.0
                    a = (mean_var - mu) / denom
                    a = 1.0 if a > 0 else 0.0

                    relapse_df.loc[relapse_df[DAY_INDEX] == day_idx, "score"] = a
                    user_preds.append(a)
                    if "relapse" in relapse_df.columns:
                        user_labels.append(relapse_df[relapse_df[DAY_INDEX]==day_idx]["relapse"].to_numpy()[0])

                if args.mode == "test":
                    save_dir = os.path.join(args.submission_path, f"patient{patient[1]}", sub)
                    os.makedirs(save_dir, exist_ok=True)
                    relapse_df.to_csv(os.path.join(save_dir, "submission.csv"), index=False)
                    print(f"Saved submission to: {save_dir}")

        # --- Print metrics for both validation and test ---
        if len(np.unique(user_labels)) > 1:
            y_true, y_pred = np.array(user_labels), np.array(user_preds)
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, y_pred)
            precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_pred)
            auroc = sklearn.metrics.auc(fpr, tpr)
            auprc = sklearn.metrics.auc(recall, precision)
            print(f"USER {patient}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, AVG={(auroc+auprc)/2:.4f}")
            all_auroc.append(auroc)
            all_auprc.append(auprc)
        else:
            print(f"USER {patient}: skipped metrics (labels missing or constant).")

    # ---- FINAL AGGREGATE TOTALS (MACRO) ----
    if all_auroc and all_auprc:
        total_auroc = float(np.mean(all_auroc))
        total_auprc = float(np.mean(all_auprc))
        total_avg = (total_auroc + total_auprc) / 2.0
        print(f"TOTAL AUROC={total_auroc:.4f}, TOTAL AUPRC={total_auprc:.4f}, AVG={total_avg:.4f}")

if __name__ == "__main__":
    main()

