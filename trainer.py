import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sklearn.metrics
from sklearn.covariance import EllipticEnvelope
import os

class Trainer:
    """Class to train the classifier"""

    def __init__(self, model, optim, sched, loaders, args):
        self.model = model
        self.optim = optim
        self.sched = sched
        self.dataloaders = loaders
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.current_best_score = 0

    def train(self):
        for epoch in range(self.args.epochs):
            epoch_metrics = {}

            # -------- Training -------- #
            self.model.train()
            torch.set_grad_enabled(True)

            for batch in tqdm(self.dataloaders['train'], desc=f'Train, {epoch}/{self.args.epochs}'):
                if batch is None:
                    continue

                x = batch['data'].to(self.args.device)        # shape: [batch, seq_len, features]
                user_ids = batch['user_id'].to(self.args.device)

                # Forward pass
                logits, features = self.model(x)  # features = pooled output from LSTM+CNN

                # Compute accuracy
                output_probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(output_probabilities, dim=1)
                acc = (predicted_class == user_ids).float().mean()

                # Compute loss and backprop
                loss = self.criterion(logits, user_ids)
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optim.step()

                # Log metrics
                for k, v in {'loss': loss.item(), 'acc': acc.item()}.items():
                    epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]

            self.sched.step()

            # -------- Feature Extraction for Elliptic Envelope -------- #
            self.model.eval()
            torch.set_grad_enabled(False)

            print('Getting features from train distribution...')
            all_features_train = []
            all_labels_train = []

            for batch in tqdm(self.dataloaders['train_distribution'], desc=f'Train Dist, {epoch}/{self.args.epochs}'):
                if batch is None:
                    continue
                x = batch['data'].to(self.args.device)
                user_ids = batch['user_id'].to(self.args.device)
                _, features = self.model(x)
                all_features_train.append(features.detach().cpu())
                all_labels_train.append(user_ids.detach().cpu())

            all_features_train = torch.vstack(all_features_train).numpy()
            all_labels_train = torch.hstack(all_labels_train).numpy()

            print('Training Robust Covariance...')
            clfs = []
            for subject in range(self.args.num_patients):
                subject_features_train = all_features_train[all_labels_train==subject]
                clf = EllipticEnvelope(support_fraction=1.0).fit(subject_features_train)
                clfs.append(clf)

            # -------- Validation -------- #
            print('Calculating accuracy on validation set and anomaly scores...')
            anomaly_scores = []
            relapse_labels = []
            user_ids_list = []

            for batch in tqdm(self.dataloaders['val'], desc=f'Val, {epoch}/{self.args.epochs}'):
                if batch is None:
                    continue
                x = batch['data'].to(self.args.device)
                user_id = batch['user_id'].item()

                # Forward
                logits, features = self.model(x)
                features = features.detach().cpu().numpy()
                current_clf = clfs[user_id]

                anomaly_score = -current_clf.decision_function(features).mean()
                anomaly_scores.append(anomaly_score)
                relapse_labels.append(batch['relapse_label'].item())
                user_ids_list.append(user_id)

            anomaly_scores = np.array(anomaly_scores)
            relapse_labels = np.array(relapse_labels)
            user_ids_list = np.array(user_ids_list)

            # -------- Metrics -------- #
            all_auroc, all_auprc = [], []

            for user in range(self.args.num_patients):
                user_anomaly_scores = anomaly_scores[user_ids_list==user]
                user_relapse_labels = relapse_labels[user_ids_list==user]

                precision, recall, _ = sklearn.metrics.precision_recall_curve(user_relapse_labels, user_anomaly_scores)
                fpr, tpr, _ = sklearn.metrics.roc_curve(user_relapse_labels, user_anomaly_scores)

                auroc = sklearn.metrics.auc(fpr, tpr)
                auprc = sklearn.metrics.auc(recall, precision)

                all_auroc.append(auroc)
                all_auprc.append(auprc)
                print(f'USER: {user}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}')

            total_auroc = np.mean(all_auroc)
            total_auprc = np.mean(all_auprc)
            total_avg = (total_auroc + total_auprc) / 2

            print(f'Total AUROC: {total_auroc:.4f}, Total AUPRC: {total_auprc:.4f}, Total AVG: {total_avg:.4f}, '
                  f'Train Loss: {np.mean(epoch_metrics["loss"]):.4f}, Train Acc: {np.mean(epoch_metrics["acc"]):.4f}')

            # Save best model
            if total_avg > self.current_best_score:
                self.current_best_score = total_avg
                os.makedirs(self.args.save_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.args.save_path, 'best_model.pth'))
                print('Saved best model!')

