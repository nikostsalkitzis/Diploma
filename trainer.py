import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sklearn.metrics
from sklearn.covariance import EllipticEnvelope
import os

class Trainer:

    ''' Class to train the CNN-LSTM classifier + anomaly detector '''

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
            print(f"\nEpoch {epoch+1}/{self.args.epochs}")

            # --- Training Phase --- #
            epoch_metrics = {}
            self.model.train()
            torch.set_grad_enabled(True)

            for batch in tqdm(self.dataloaders['train'], desc=f"Train {epoch+1}"):
                if batch is None:
                    continue

                x = batch['data'].to(self.args.device)      # shape: (B, C, T)
                user_ids = batch['user_id'].to(self.args.device)

                # Forward
                logits, features = self.model(x)

                # Accuracy
                predicted_class = torch.argmax(torch.softmax(logits, dim=1), dim=1)
                acc = (predicted_class == user_ids).float().mean()

                # Loss + backprop
                loss = self.criterion(logits, user_ids)
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                self.optim.step()

                # Logging
                for k, v in {"loss": loss.item(), "acc": acc.item()}.items():
                    epoch_metrics.setdefault(k, []).append(v)

            self.sched.step()

            # --- Validation Phase --- #
            self.model.eval()
            torch.set_grad_enabled(False)

            print("Getting features from train distribution...")
            all_features_train, all_labels_train = [], []

            for batch in tqdm(self.dataloaders["train_distribution"], desc="Train distribution features"):
                if batch is None:
                    continue

                x = batch["data"].to(self.args.device)
                user_ids = batch["user_id"].to(self.args.device)

                _, features = self.model(x)
                all_features_train.append(features.detach().cpu())
                all_labels_train.append(user_ids.detach().cpu())

            all_features_train = torch.vstack(all_features_train).numpy()
            all_labels_train = torch.hstack(all_labels_train).numpy()

            # Train one EllipticEnvelope per patient
            print("Training EllipticEnvelope for outlier detection...")
            clfs = []
            for subject in range(self.args.num_patients):
                subject_features = all_features_train[all_labels_train == subject]
                clf = EllipticEnvelope(support_fraction=1.0).fit(subject_features)
                clfs.append(clf)

            print("Calculating accuracy on validation set and anomaly scores...")
            anomaly_scores, relapse_labels, user_ids_list = [], [], []

            for batch in tqdm(self.dataloaders["val"], desc="Validation"):
                if batch is None:
                    continue

                x = batch["data"].to(self.args.device)   # may be (1, N, C, T) or (B, C, T)
                user_id = batch["user_id"].to(self.args.device)

                if x.dim() == 4:
                    # flatten to (N, C, T)
                    x = x.squeeze(0)

                _, features = self.model(x)

                # if multiple sequences per day, average them
                if features.dim() > 2:   # (N, hidden_size)
                    features = features.mean(dim=0, keepdim=True)
                elif features.dim() == 1:   # (hidden_size,)
                    features = features.unsqueeze(0)

                features_np = features.detach().cpu().numpy()
                current_clf = clfs[user_id.item()]

                anomaly_score = -current_clf.decision_function(features_np).mean()
                anomaly_scores.append(anomaly_score)
                relapse_labels.append(batch["relapse_label"].item())
                user_ids_list.append(batch["user_id"].item())

            anomaly_scores = np.array(anomaly_scores)
            relapse_labels = np.array(relapse_labels)
            user_ids_list = np.array(user_ids_list)

            print("Calculating metrics...")
            all_auroc, all_auprc = [], []

            for user in range(self.args.num_patients):
                user_scores = anomaly_scores[user_ids_list == user]
                user_labels = relapse_labels[user_ids_list == user]

                if len(np.unique(user_labels)) < 2:
                    continue

                precision, recall, _ = sklearn.metrics.precision_recall_curve(user_labels, user_scores)
                fpr, tpr, _ = sklearn.metrics.roc_curve(user_labels, user_scores)

                auroc = sklearn.metrics.auc(fpr, tpr)
                auprc = sklearn.metrics.auc(recall, precision)
                all_auroc.append(auroc)
                all_auprc.append(auprc)

                print(f"USER: {user}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}")

            total_auroc = np.mean(all_auroc) if all_auroc else 0.0
            total_auprc = np.mean(all_auprc) if all_auprc else 0.0
            total_avg = (total_auroc + total_auprc) / 2

            print(
                f"Epoch {epoch+1} Summary: "
                f"Train Loss={np.mean(epoch_metrics['loss']):.4f}, "
                f"Train Acc={np.mean(epoch_metrics['acc']):.4f}, "
                f"Val AUROC={total_auroc:.4f}, "
                f"Val AUPRC={total_auprc:.4f}, "
                f"Avg={total_avg:.4f}"
            )

            # Save best model
            if total_avg > self.current_best_score:
                self.current_best_score = total_avg
                os.makedirs(self.args.save_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.args.save_path, "best_model.pth"))
                print("Saved best model!")
