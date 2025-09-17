import torch
import argparse
from torch.optim.lr_scheduler import MultiStepLR
from model import CNNLSTMClassifier
from dataset import PatientDataset
from trainer import Trainer
import pickle
import os

def parse():
    '''Returns args passed to the train.py script.'''
    parser = argparse.ArgumentParser()

    # LSTM+CNN parameters
    parser.add_argument('--window_size', type=int, default=48)
    parser.add_argument('--input_features', type=int, default=8)
    parser.add_argument('--cnn_channels', type=int, default=128)
    parser.add_argument('--lstm_hidden', type=int, default=32)
    parser.add_argument('--lstm_layers', type=int, default=1)

    # num_patients
    parser.add_argument('--num_patients', type=int, default=8)

    # input paths
    parser.add_argument('--features_path', type=str, required=True, help='features path')
    parser.add_argument('--dataset_path', type=str, required=True, help='dataset path for relapse labels')

    # learning params    
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)

    # checkpoint
    parser.add_argument('--save_path', type=str, default='checkpoints')

    # device
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    return args

def main():
    args = parse()
    device = args.device
    print('Using device:', device)

    # Model
    model = CNNLSTMClassifier(
        input_features=args.input_features,
        cnn_channels=args.cnn_channels,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        window_size=args.window_size,
        num_patients=args.num_patients,
        device=device
    )
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', n_parameters)

    # Optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9,0.999), weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer, milestones=[args.epochs//2, args.epochs//4*3], gamma=0.1)

    # Dataset
    train_dataset = PatientDataset(
        features_path=args.features_path,
        dataset_path=args.dataset_path,
        mode='train',
        window_size=args.window_size
    )

    # Save scaler
    os.makedirs(args.save_path, exist_ok=True)
    with open(os.path.join(args.save_path, 'scaler.pkl'), 'wb') as f:
        pickle.dump(train_dataset.scaler, f)

    valid_dataset = PatientDataset(
        features_path=args.features_path,
        dataset_path=args.dataset_path,
        mode='val',
        scaler=train_dataset.scaler,
        window_size=args.window_size
    )

    print('Length of train dataset:', len(train_dataset))
    print('Length of valid dataset:', len(valid_dataset))

    # Collate function to ignore None
    def collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    loaders = {
        'train': torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, collate_fn=collate_fn
        ),
        'val': torch.utils.data.DataLoader(
            valid_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn
        ),
        'train_distribution': torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=True, collate_fn=collate_fn
        )
    }

    # Trainer
    trainer = Trainer(
        model=model,
        optim=optimizer,
        sched=scheduler,
        loaders=loaders,
        args=args
    )

    trainer.train()


if __name__ == '__main__':
    main()
