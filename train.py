import torch
import argparse
from torch.optim.lr_scheduler import MultiStepLR
from model import LSTMCNNClassifier
from dataset import PatientDataset
from trainer import Trainer
import pickle
import os

def parse():
    parser = argparse.ArgumentParser()

    # LSTM+CNN parameters
    parser.add_argument('--input_features', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--cnn_channels', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.2)

    parser.add_argument('--num_patients', type=int, default=2)
    parser.add_argument('--features_path', type=str, help='features to use')
    parser.add_argument('--dataset_path', type=str, help='dataset for relapse labels')

    # learning parameters
    parser.add_argument('--optimizer', type=str, choices=['SGD','Adam'], default='Adam')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=10)

    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()
    args.seq_len = args.window_size  # optional, for compatibility
    return args

def main():
    args = parse()
    device = args.device
    print('Using device', device)

    # Model
    model = LSTMCNNClassifier(**vars(args))
    
    model.to(device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters:', n_parameters)

    # Optimizer
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = MultiStepLR(optimizer, milestones=[args.epochs//2, args.epochs//4*3], gamma=0.1)

    # Datasets
    train_dataset = PatientDataset(features_path=args.features_path, 
                                   dataset_path=args.dataset_path,
                                   mode='train', window_size=args.window_size)

    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/scaler.pkl', 'wb') as f:
        pickle.dump(train_dataset.scaler, f)

    valid_dataset = PatientDataset(features_path=args.features_path,
                                   dataset_path=args.dataset_path,
                                   mode='val', scaler=train_dataset.scaler,
                                   window_size=args.window_size)

    print('Length of train dataset:', len(train_dataset))
    print('Length of valid dataset:', len(valid_dataset))

    def collate_fn(batch):
        batch = [x for x in batch if x is not None]
        if len(batch) == 0:
            return None
        return torch.utils.data.dataloader.default_collate(batch)

    loaders = {
        'train': torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                             num_workers=8, pin_memory=True, collate_fn=collate_fn),
        'val': torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                           num_workers=8, pin_memory=True, collate_fn=collate_fn),
        'train_distribution': torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
                                                          num_workers=8, pin_memory=True, collate_fn=collate_fn)
    }

    trainer = Trainer(model=model, optim=optimizer, sched=scheduler, loaders=loaders, args=args)
    trainer.train()

if __name__ == '__main__':
    main()

