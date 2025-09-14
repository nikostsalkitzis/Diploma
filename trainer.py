import os
import numpy as np
import torch
import torch.nn as nn
import sklearn.metrics
from sklearn.covariance import EllipticEnvelope
from tqdm import tqdm


class Trainer:
    """Class to train the classifier."""

    def __init__(self, model, optim, sched, loaders, args):
        self.model = model
        self.optim = optim
        self.sched = sched
        self.dataloaders = loaders
        self.args = args
        self.criterion = nn.CrossEntropyLoss()
        self.current_best_score = 0.0

    def train(self):
        for epoch in range(self.args.epochs):
            epoch_metrics = {"loss": [], "acc": []}

            # -------------------- TRAINING -------------------- #
            self.model.train()
            torch.set_grad_enabled(True)

            for batch in tqdm(self.dataloaders["train"], desc=f"Train {epoch}/{self.args.epochs-1}"):
                if batch is None:
                    continue

                x = batch["data"].to(self.args.device)       # [batch, seq_len, features]
                y = batch["user_id"].to(self.args.device)    # [batch]

                logits, features = self.model(x)

                # Loss + backprop
                loss = self.criterion(logits, y)
                self.optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optim.step()

                # Accuracy
                preds = torch.argmax(logits, dim=1)
                acc = (preds == y).float().mean()

                epoch_metrics["loss"].append(loss.item())
                epoch_metrics["acc"].append(acc.item())

            # scheduler after each epoch
            if self.sched is not None:
                self.sched.step()

            # -------------------- FEATURE EXTRACTION -------------------- #
            self.model.eval()
            torch.set_grad_enabled(False)

            print("Extracting features from train distribution...")
            feats_train, labels_train = [], []

            for batch in tqdm(self.dataloaders["train_distribution"], desc=f"TrainDist {epoch}"):
                if batch is None:
                    continue

                x = batch["data"].to(self.args.device)
                y = batch["user_id"].to(self.args.device)
                _, f = self.model(x)

                feats_train.append(f.detach().cpu())
                labels_train.append(y.detach().cpu())

            if len(feats_train) == 0:
                print("⚠️ No features collected from train_distribution. Skipping envelope fit.")
                continue

            feats_train = torch.vstack(feats_train).numpy()
            labels_train = torch.hstack(labels_train).numpy()

            # Fit EllipticEnvelope for each patient
            print("Training robust covariance models...")
            clfs = []
            for subject in range(self.args.num_patients):
                subj_feats = feats_train[labels_train == subject]
                if len(subj_feats) < 2:
                    # fallback: identity model if not enough samples
                    clfs.append(None)
                    continue
                clf = EllipticEnvelope(support_fraction=1.0)
                clf.fit(subj_feats)
                clfs.append(clf)

            # -------------------- VALIDATION -------------------- #
            print("Validation & anomaly scoring...")
            anomaly_scores, relapse_labels, user_ids = [], [], []

            for batch in tqdm(self.dataloaders["val"], desc=f"Val {epoch}"):
                if batch is None:
                    continue

                x = batch["data"].to(self.args.device)
                y_user = batch["user_id"].item()
                relapse = batch["relapse_label"].item()

                logits, f = self.model(x)
                f = f.detach().cpu().numpy()

                clf = clfs[y_user]
                if clf is None:
                    score = 0.0  # if no model for that user
                else:
                    score = -clf.decision_function(f).mean()

                anomaly_scores.append(score)
                relapse_labels.append(relapse)
                user_ids.append(y_user)

            if len(anomaly_scores) == 0:
                print("⚠️ Validation set is empty.")
                continue

            anomaly_scores = np.array(anomaly_scores)
            relapse_labels = np.array(relapse_labels)
            user_ids = np.array(user_ids)

            # -------------------- METRICS -------------------- #
            aurocs, auprcs = [], []

            for u in range(self.args.num_patients):
                mask = user_ids == u
                if mask.sum() == 0:
                    continue
                scores_u = anomaly_scores[mask]
                labels_u = relapse_labels[mask]

                precision, recall, _ = sklearn.metrics.precision_recall_curve(labels_u, scores_u)
                fpr, tpr, _ = sklearn.metrics.roc_curve(labels_u, scores_u)

                auroc = sklearn.metrics.auc(fpr, tpr)
                auprc = sklearn.metrics.auc(recall, precision)

                aurocs.append(auroc)
                auprcs.append(auprc)
                print(f"User {u}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

            if len(aurocs) == 0:
                print("⚠️ No AUROC/AUPRC could be computed.")
                continue

            total_auroc = np.mean(aurocs)
            total_auprc = np.mean(auprcs)
            total_avg = 0.5 * (total_auroc + total_auprc)

            print(
                f"Total AUROC: {total_auroc:.4f} | "
                f"Total AUPRC: {total_auprc:.4f} | "
                f"AVG: {total_avg:.4f} | "
                f"TrainLoss: {np.mean(epoch_metrics['loss']):.4f} | "
                f"TrainAcc: {np.mean(epoch_metrics['acc']):.4f}"
            )

            # -------------------- SAVE BEST MODEL -------------------- #
            if total_avg > self.current_best_score:
                self.current_best_score = total_avg
                os.makedirs(self.args.save_path, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(self.args.save_path, "best_model.pth"))
                print("✅ Saved new best model.")
