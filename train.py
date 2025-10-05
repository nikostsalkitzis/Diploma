import torch
import argparse
from torch.optim.lr_scheduler import MultiStepLR
from model import CNNLSTMClassifier        # <── Updated import
from dataset import PatientDataset
from trainer import Trainer
import pickle
import os


# ------------------------------------------------------------
# Utility: choose device
# ------------------------------------------------------------
def get_device(device_str="auto") -> str:
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    else:
        try:
            torch.device(device_str)
            return device_str
        except Exception as e:
            print("Invalid device; falling back to CPU:", e)
            return "cpu"


# ------------------------------------------------------------
# Argument parser
# ------------------------------------------------------------
def parse():
    parser = argparse.ArgumentParser()

    # System / environment
    parser.add_argument('--cores', type=int, default=os.cpu_count())
    parser.add_argument('--ensembles', type=int, default=5)

    # Model hyperparameters
    parser.add_argument('--window_size', type=int, default=24)
    parser.add_argument('--stride', type=int, default=12)
    parser.add_argument('--input_features', type=int, default=8)
    parser.add_argument('--output_dim', type=int, default=2)
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)

    # Patients
    parser.add_argument('--num_patients', type=int, default=8)

    # Dataset paths
    parser.add_argument('--features_path', default="data/track_2_new_features/", type=str,
                        help='path to preprocessed patient features')
    parser.add_argument('--dataset_path', default="data/track_2/", type=str,
                        help='path to relapse labels')

    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=50)

    # Checkpoint path
    parser.add_argument('--save_path', type=str, default='checkpoints')

    # Device
    default_device = get_device()
    parser.add_argument('--device', type=str, default=default_device)

    args = parser.parse_args()
    args.seq_len = args.window_size
    return args


# ------------------------------------------------------------
# Main training script
# ------------------------------------------------------------
def main():
    # Parse args
    args = parse()
    device = args.device
    print('Using device:', device)

    # -------------------------------
    # 1️⃣ Create models (one per patient)
    # -------------------------------
    models = [CNNLSTMClassifier(vars(args)).to(device) for _ in range(args.num_patients)]

    n_parameters = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    print('Number of trainable parameters:', n_parameters)

    # -------------------------------
    # 2️⃣ Create optimizers & schedulers
    # -------------------------------
    optimizers = [
        torch.optim.Adam(
            params=models[i].parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
        for i in range(args.num_patients)
    ]

    schedulers = [
        MultiStepLR(optimizers[i],
                    milestones=[args.epochs // 2, args.epochs // 4 * 3],
                    gamma=0.1)
        for i in range(args.num_patients)
    ]

    # -------------------------------
    # 3️⃣ Create datasets and dataloaders
    # -------------------------------
    train_datasets, train_dist_datasets, valid_datasets = [], [], []

    for patient in [f"P{i}" for i in range(1, args.num_patients + 1)]:
        print(f"Loading patient {patient} ...")

        # Training dataset
        train_dataset = PatientDataset(
            features_path=args.features_path,
            dataset_path=args.dataset_path,
            mode='train',
            window_size=args.window_size,
            stride=args.stride,
            patient=patient
        )
        train_datasets.append(train_dataset)

        # Save scaler for reproducibility
        patient_id = patient[1:]
        patient_path = os.path.join(args.save_path, patient_id)
        os.makedirs(patient_path, exist_ok=True)
        with open(f'{patient_path}/scaler.pkl', 'wb') as f:
            pickle.dump(train_dataset.scaler, f)

        # Validation dataset (same scaler)
        valid_dataset = PatientDataset(
            features_path=args.features_path,
            dataset_path=args.dataset_path,
            mode='val',
            scaler=train_dataset.scaler,
            window_size=args.window_size,
            stride=args.stride,
            patient=patient
        )
        valid_datasets.append(valid_dataset)

        # Train-distance dataset (used for ensemble uncertainty calibration)
        train_dist_dataset = PatientDataset(
            features_path=args.features_path,
            dataset_path=args.dataset_path,
            mode='train',
            window_size=args.window_size,
            stride=args.stride,
            patient=patient
        )
        train_dist_datasets.append(train_dist_dataset)

    # Dataloaders per patient
    all_loaders = []
    for i in range(args.num_patients):
        loaders = {
            'train': torch.utils.data.DataLoader(
                train_datasets[i],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cores,
                pin_memory=True
            ),
            'val': torch.utils.data.DataLoader(
                valid_datasets[i],
                batch_size=1,
                shuffle=False,
                num_workers=args.cores,
                pin_memory=True
            ),
            'train_dist': torch.utils.data.DataLoader(
                train_dist_datasets[i],
                batch_size=1,
                shuffle=False,
                num_workers=args.cores,
                pin_memory=True
            )
        }
        all_loaders.append(loaders)

    # -------------------------------
    # 4️⃣ Initialize trainer and start training
    # -------------------------------
    trainer = Trainer(models, optimizers, schedulers, all_loaders, args)
    trainer.train()


if __name__ == '__main__':
    main()
