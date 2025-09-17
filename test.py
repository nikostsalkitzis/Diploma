import torch
import argparse
from model import CNNLSTMClassifier
from dataset import PatientDataset
import pickle
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_curve, precision_recall_curve, auc

def parse():
    parser = argparse.ArgumentParser()

    # model parameters
    parser.add_argument('--window_size', type=int, default=48)
    parser.add_argument('--input_features', type=int, default=8)
    parser.add_argument('--cnn_channels', type=int, default=16)
    parser.add_argument('--lstm_hidden', type=int, default=32)
    parser.add_argument('--lstm_layers', type=int, default=1)

    # num patients
    parser.add_argument('--num_patients', type=int, default=10)

    # paths
    parser.add_argument('--features_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--submission_path', type=str, required=True)

    # checkpoint & scaler
    parser.add_argument('--load_path', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--scaler_path', type=str, default='checkpoints/scaler.pkl')

    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mode', type=str, choices=['val', 'test'], default='test')

    return parser.parse_args()


def main():
    args = parse()
    device = args.device
    window_size = args.window_size
    print('Using device:', device)

    # Features
    columns_to_scale = ['acc_norm', 'heartRate_mean', 'rRInterval_mean', 'rRInterval_rmssd',
                        'rRInterval_sdnn', 'rRInterval_lombscargle_power_high']
    data_columns = columns_to_scale + ['sin_t', 'cos_t']

    # Load model
    model = CNNLSTMClassifier(
        input_features=args.input_features,
        cnn_channels=args.cnn_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        num_patients=args.num_patients
    )
    model.to(device)

    checkpoint = torch.load(args.load_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    torch.set_grad_enabled(False)

    # Load scaler
    with open(args.scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    # Collate fn
    def collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    # Train dataset to extract features for anomaly detection
    train_dataset = PatientDataset(
        features_path=args.features_path,
        dataset_path=args.dataset_path,
        mode='train',
        window_size=window_size
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=False,
        num_workers=8, pin_memory=True, collate_fn=collate_fn
    )

    # Collect features from training data
    all_features_train = []
    all_labels_train = []

    for batch in tqdm(train_loader, desc='Extracting train features'):
        if batch is None:
            continue

        x = batch['data'].to(device)
        user_ids = batch['user_id'].to(device)

        _, features = model(x)
        all_features_train.append(features.detach().cpu())
        all_labels_train.append(user_ids.detach().cpu())

    all_features_train = torch.vstack(all_features_train).numpy()
    all_labels_train = torch.hstack(all_labels_train).numpy()

    # Train Elliptic Envelope per user
    clfs = []
    for user in range(args.num_patients):
        user_features = all_features_train[all_labels_train == user]
        clf = EllipticEnvelope(support_fraction=1.0).fit(user_features)
        clfs.append(clf)

    # Evaluate per patient
    all_auroc = []
    all_auprc = []
    random_auroc = []
    random_auprc = []

    for patient in os.listdir(args.features_path):
        patient_dir = os.path.join(args.features_path, patient)

        user_relapse_labels = []
        user_anomaly_scores = []

        for subfolder in os.listdir(patient_dir):
            if (args.mode == 'val' and 'val' in subfolder) or (args.mode == 'test' and 'test' in subfolder):
                subfolder_dir = os.path.join(patient_dir, subfolder)
                file_path = os.path.join(subfolder_dir, 'features.csv')
                df = pd.read_csv(file_path, index_col=0)
                df = df.replace([np.inf, -np.inf], np.nan).dropna()

                relapse_df = pd.read_csv(os.path.join(args.dataset_path, patient, subfolder, 'relapses.csv'))
                relapse_df = relapse_df.iloc[:-1]  # drop last row if falsely added

                user_id = int(patient[1:]) - 1
                anomaly_scores = []

                for day_index in relapse_df['day_index'].unique():
                    day_data = df[df['day_index'] == day_index].copy()
                    relapse_label = relapse_df[relapse_df['day_index'] == day_index]['relapse'].to_numpy()[0]

                    if len(day_data) < window_size:
                        score = 0
                    else:
                        sequences = []
                        if len(day_data) == window_size:
                            sequence = day_data[data_columns].to_numpy()
                            sequence[:, :-2] = scaler.transform(sequence[:, :-2])
                            sequences.append(sequence)
                        else:
                            for start_idx in range(0, len(day_data) - window_size, window_size // 3):
                                sequence = day_data.iloc[start_idx:start_idx + window_size][data_columns].to_numpy()
                                sequence[:, :-2] = scaler.transform(sequence[:, :-2])
                                sequences.append(sequence)

                        sequence_tensor = torch.tensor(np.stack(sequences), dtype=torch.float32).permute(0, 2, 1).to(device)
                        _, features = model(sequence_tensor)
                        features = features.detach().cpu().numpy()

                        score = -clfs[user_id].decision_function(features).mean()

                    anomaly_scores.append(score)
                    relapse_df.loc[relapse_df['day_index'] == day_index, 'anomaly_score'] = score
                    user_anomaly_scores.append(score)
                    user_relapse_labels.append(relapse_label)

                # Save submission
                save_dir = os.path.join(args.submission_path, patient, subfolder)
                os.makedirs(save_dir, exist_ok=True)
                relapse_df.to_csv(os.path.join(save_dir, 'submission.csv'), index=False)

                print(f'Patient {patient} | Subfolder {subfolder} processed')

        # Compute AUROC / AUPRC if in validation mode
        if args.mode != 'test':
            user_anomaly_scores = np.array(user_anomaly_scores)
            user_relapse_labels = np.array(user_relapse_labels)

            fpr, tpr, _ = roc_curve(user_relapse_labels, user_anomaly_scores)
            precision, recall, _ = precision_recall_curve(user_relapse_labels, user_anomaly_scores)

            auroc = auc(fpr, tpr)
            auprc = auc(recall, precision)

            all_auroc.append(auroc)
            all_auprc.append(auprc)
            random_auroc.append(0.5)
            random_auprc.append(user_relapse_labels.mean())

            print(f'USER: {patient}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, Random AUPRC: {user_relapse_labels.mean():.4f}')

    # Total AUROC/AUPRC
    if args.mode != 'test' and len(all_auroc) > 0:
        total_auroc = sum(all_auroc) / len(all_auroc)
        total_auprc = sum(all_auprc) / len(all_auprc)
        total_avg = (total_auroc + total_auprc) / 2
        print(f'Total AUROC: {total_auroc:.4f}, Total AUPRC: {total_auprc:.4f}, Total AVG: {total_avg:.4f}, Random AUROC: 0.5000, Random AUPRC: {sum(random_auprc)/len(random_auprc):.4f}, Ideal AVG: {(0.5 + sum(random_auprc)/len(random_auprc))/2:.4f}')


if __name__ == '__main__':
    main()
