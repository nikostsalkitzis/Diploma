import torch
import argparse
from model import LSTMCNNClassifier
from dataset import PatientDataset
import pickle
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.covariance import EllipticEnvelope
import sklearn.metrics

def parse():
    parser = argparse.ArgumentParser()

    # LSTM+CNN parameters
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--cnn_channels', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)

    # num_patients
    parser.add_argument('--num_patients', type=int, default=10)

    # input paths
    parser.add_argument('--features_path', type=str, help='features to use')
    parser.add_argument('--dataset_path', type=str, help='dataset path')
    parser.add_argument('--submission_path', type=str, help='where to save submissions')

    # checkpoint
    parser.add_argument('--load_path', type=str, help='path to saved model', default='checkpoints/best_model.pth')
    parser.add_argument('--scaler_path', type=str, help='path to saved scaler', default='checkpoints/scaler.pkl')

    parser.add_argument('--device', type=str, help='device to use (cpu, cuda, cuda[number])', default='cuda')
    parser.add_argument('--mode', type=str, help='val or test', default='test')

    parser.add_argument('--window_size', type=int, default=48)

    args = parser.parse_args()
    args.seq_len = args.window_size
    return args


def main():
    args = parse()
    device = args.device
    window_size = args.window_size
    print('Using device', device)

    # ------------------ Load Model ------------------ #
    model = LSTMCNNClassifier(vars(args))
    model.to(device)
    checkpoint = torch.load(args.load_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    torch.set_grad_enabled(False)

    # ------------------ Load Scaler ------------------ #
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # ------------------ Dataset Loader ------------------ #
    def collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    train_dataset = PatientDataset(
        features_path=args.features_path,
        dataset_path=args.dataset_path,
        mode='train',
        window_size=window_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # ------------------ Extract Features from Train ------------------ #
    print('Extracting features from training set...')
    all_features_train = []
    all_labels_train = []

    for batch in tqdm(train_loader):
        if batch is None:
            continue
        x = batch['data'].to(device)
        user_ids = batch['user_id'].to(device)

        _, features = model(x)  # pooled features from LSTM+CNN
        all_features_train.append(features.detach().cpu())
        all_labels_train.append(user_ids.detach().cpu())

    all_features_train = torch.vstack(all_features_train).numpy()
    all_labels_train = torch.hstack(all_labels_train).numpy()

    # ------------------ Train Elliptic Envelopes ------------------ #
    print('Training Elliptic Envelope per patient...')
    clfs = []
    for subject in range(args.num_patients):
        subject_features = all_features_train[all_labels_train == subject]
        clf = EllipticEnvelope(support_fraction=1.0).fit(subject_features)
        clfs.append(clf)

    # ------------------ Iterate Patients and Compute Anomaly Scores ------------------ #
    all_auroc = []
    all_auprc = []

    for patient in os.listdir(args.features_path):
        patient_dir = os.path.join(args.features_path, patient)
        user_anomaly_scores = []
        user_relapse_labels = []

        for subfolder in os.listdir(patient_dir):
            if (args.mode == 'val' and 'val' in subfolder) or (args.mode == 'test' and 'test' in subfolder):
                subfolder_dir = os.path.join(patient_dir, subfolder)
                features_file = os.path.join(subfolder_dir, 'features.csv')
                df = pd.read_csv(features_file, index_col=0).replace([np.inf, -np.inf], np.nan).dropna()

                relapse_df = pd.read_csv(os.path.join(args.dataset_path, patient, subfolder, 'relapses.csv'))
                relapse_df = relapse_df.iloc[:-1]  # remove last false row

                day_indices = relapse_df['day_index'].unique()
                patient_id = int(patient[1:]) - 1

                for day_index in day_indices:
                    day_data = df[df['day_index'] == day_index].copy()
                    relapse_label = relapse_df[relapse_df['day_index'] == day_index]['relapse'].to_numpy()[0]

                    if len(day_data) < window_size:
                        anomaly_score = 0
                        user_anomaly_scores.append(anomaly_score)
                        user_relapse_labels.append(relapse_label)
                        relapse_df.loc[relapse_df['day_index'] == day_index, 'anomaly_score'] = anomaly_score
                        continue

                    sequences = []
                    for start_idx in range(0, len(day_data) - window_size + 1, window_size // 3):
                        sequence = day_data.iloc[start_idx:start_idx + window_size].copy()
                        seq_array = sequence.to_numpy()
                        seq_array[:, :-2] = scaler.transform(seq_array[:, :-2])
                        sequences.append(seq_array)

                    sequence_tensor = torch.tensor(np.stack(sequences), dtype=torch.float32).to(device)
                    # LSTM+CNN expects [batch, seq_len, features]
                    logits, features = model(sequence_tensor)
                    features = features.detach().cpu().numpy()
                    anomaly_score = -clfs[patient_id].decision_function(features).mean()
                    user_anomaly_scores.append(anomaly_score)
                    user_relapse_labels.append(relapse_label)
                    relapse_df.loc[relapse_df['day_index'] == day_index, 'anomaly_score'] = anomaly_score

                # Save submission CSV
                os.makedirs(os.path.join(args.submission_path, patient, subfolder), exist_ok=True)
                relapse_df.to_csv(os.path.join(args.submission_path, patient, subfolder, 'submission.csv'))

        # Compute AUROC/AUPRC per patient if in validation mode
        if args.mode != 'test':
            user_anomaly_scores = np.array(user_anomaly_scores)
            user_relapse_labels = np.array(user_relapse_labels)
            precision, recall, _ = sklearn.metrics.precision_recall_curve(user_relapse_labels, user_anomaly_scores)
            fpr, tpr, _ = sklearn.metrics.roc_curve(user_relapse_labels, user_anomaly_scores)
            auroc = sklearn.metrics.auc(fpr, tpr)
            auprc = sklearn.metrics.auc(recall, precision)
            all_auroc.append(auroc)
            all_auprc.append(auprc)
            print(f'USER: {patient}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}')

    # Print total metrics
    if args.mode != 'test' and all_auroc:
        total_auroc = np.mean(all_auroc)
        total_auprc = np.mean(all_auprc)
        total_avg = (total_auroc + total_auprc) / 2
        print(f'Total AUROC: {total_auroc:.4f}, Total AUPRC: {total_auprc:.4f}, Total AVG: {total_avg:.4f}')


if __name__ == '__main__':
    main()

