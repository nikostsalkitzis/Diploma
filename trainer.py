import pickle
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sklearn.metrics
from model import EnsembleLinear


# -------------------------------------------------------
# Create Ensemble MLP (unchanged core, adapted for heart targets)
# -------------------------------------------------------
def create_ensemble_mlp(args):
    m = nn.Sequential(
        EnsembleLinear(args.d_model, args.d_model, args.ensembles),
        nn.ReLU(),
        EnsembleLinear(args.d_model, args.d_model, args.ensembles),
        nn.ReLU(),
        EnsembleLinear(args.d_model, args.output_dim, args.ensembles),
    )
    m.to(args.device)
    return m


# -------------------------------------------------------
# Trainer Class
# -------------------------------------------------------
class Trainer:
    """ Trainer for TransformerHeartPredictor + Ensemble anomaly modeling """

    def __init__(self, models, optims, scheds, loaders, args):
        self.models = models
        self.optims = optims
        self.scheds = scheds
        self.dataloaders = loaders
        self.args = args

        # Loss function (regression only)
        self.regression_loss = nn.MSELoss()

        # Track best metrics
        self.current_best_avgs = [-np.inf for _ in range(len(self.models))]
        self.current_best_aurocs = [-np.inf for _ in range(len(self.models))]
        self.current_best_auprcs = [-np.inf for _ in range(len(self.models))]

        # Ensemble heads for anomaly detection
        self.mlps = []
        self.optimizers = []
        for _ in range(len(self.models)):
            ensemble_head = create_ensemble_mlp(self.args)
            self.mlps.append(ensemble_head)
            ps = ensemble_head.parameters()
            self.optimizers.append(torch.optim.Adam(ps, lr=1e-3, weight_decay=1e-4))

    # ---------------------------------------------------
    # Stage 1: Train the Transformer encoder (heart prediction)
    # ---------------------------------------------------
    def train_encoder_once(self, epoch: int, i: int, epoch_metrics: dict):
        """ Train Transformer encoder on heart feature prediction """
        self.models[i].train()
        for batch in tqdm(self.dataloaders[i]["train"], desc=f"Train Encoder ({i+1}/{len(self.models)})"):
            if batch is None:
                continue

            x = batch["data"].to(self.args.device)
            heart_labels = batch["target"].to(self.args.device)  # (batch, 5, 1)
            heart_labels = torch.squeeze(heart_labels, dim=-1)   # (batch, 5)

            # Forward pass
            features, heart_preds = self.models[i](x)

            # Loss
            loss_heart = self.regression_loss(heart_preds, heart_labels)
            total_loss = loss_heart

            # Backward
            self.optims[i].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models[i].parameters(), 5)
            self.optims[i].step()

            # Log metrics
            metrics = {
                "loss_total": total_loss.item(),
                "loss_heart": loss_heart.item(),
            }
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, []) + [v]

        return epoch_metrics

    # ---------------------------------------------------
    # Helper: Resample batch for ensemble training
    # ---------------------------------------------------
    def resample_batch(self, indices, dataset):
        ensemble_size = self.args.ensembles
        offsets = torch.zeros_like(indices)
        data_batch = []
        target_batch = []
        for i in range(indices.size()[0]):
            while (offsets[i] % ensemble_size) == 0:
                offsets[i] = torch.randint(low=1, high=len(dataset), size=(1,))
            r_idx = (offsets[i] + indices[i]) % len(dataset)
            x = dataset[r_idx]["data"]
            t = dataset[r_idx]["target"]
            data_batch.append(x)
            target_batch.append(t)
        data = torch.stack(data_batch).to(self.args.device)
        targets = torch.stack(target_batch).to(self.args.device)
        return data, targets

    # ---------------------------------------------------
    # Stage 2: Train ensemble MLPs on heart prediction
    # ---------------------------------------------------
    def train_ensemble(self, i: int):
        """ Train ensemble MLP on top of encoder features (heart targets) """
        k = self.args.ensembles
        ensemble_head = self.mlps[i]
        ensemble_head.train()
        optim = self.optimizers[i]

        for batch in tqdm(self.dataloaders[i]["train"], desc=f"Train Ensemble ({i+1}/{len(self.models)})"):
            if batch is None:
                continue

            x = batch["data"].to(self.args.device)
            targets = batch["target"].to(self.args.device)
            targets = torch.squeeze(targets, dim=-1)

            features, heart_preds = self.models[i](x)

            batched_features = features[None, :, :].repeat([k, 1, 1])
            batched_targets = targets[None, :, :].repeat([k, 1, 1])
            ensemble_mask = batch["idx"] % k

            resampled_x, r_targets = self.resample_batch(
                batch["idx"], self.dataloaders[i]["train"].dataset
            )
            r_targets = torch.squeeze(r_targets, dim=-1)
            r_features, _ = self.models[i](resampled_x)

            batched_features[ensemble_mask] = r_features
            batched_targets[ensemble_mask] = r_targets

            mean = ensemble_head.forward(batched_features)
            mse_loss = torch.sum(torch.pow(mean - batched_targets, 2), dim=(0, 2))
            total_loss = torch.mean(mse_loss)

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble_head.parameters(), 5)
            optim.step()

    # ---------------------------------------------------
    # Evaluate ensembles (validation anomaly detection)
    # ---------------------------------------------------
    def evaluate_ensembles(self, epoch: int, i: int, train_dist_anomaly_scores: dict):
        """ Validation: anomaly detection based on heart prediction variance """
        k = self.args.ensembles
        anomaly_scores = []
        relapse_labels = []
        user_ids = []

        _mean = np.mean(train_dist_anomaly_scores[i])
        _max, _min = np.max(train_dist_anomaly_scores[i]), np.min(train_dist_anomaly_scores[i])

        for batch in self.dataloaders[i]["val"]:
            if batch is None:
                continue
            user_id = batch["user_id"].to(self.args.device)
            if user_id.item() >= self.args.num_patients:
                continue

            x = batch["data"].to(self.args.device).squeeze(0)
            targets = batch["target"].to(self.args.device)
            targets = torch.squeeze(targets)
            if targets.ndim < 2:
                targets = torch.reshape(targets, (1, -1))

            targets = targets[None, :, :].repeat([k, 1, 1])

            features, _ = self.models[i](x)
            batched_features = features[None, :, :].repeat([k, 1, 1])
            preds = self.mlps[user_id.item()].forward(batched_features)

            average_pred = torch.mean(preds, 0)
            var_score = torch.sum((preds - average_pred) ** 2, dim=(2,))
            mean_var = torch.mean(torch.mean(var_score, 0)).item()
            anomaly_score = (mean_var - _mean) / (_max - _min)

            anomaly_scores.append(anomaly_score)
            relapse_labels.append(batch["relapse_label"].item())  # still included
            user_ids.append(batch["user_id"].item())

        anomaly_scores = (np.array(anomaly_scores) > 0.0).astype(np.float64)
        relapse_labels = np.array(relapse_labels)
        user_ids = np.array(user_ids)
        return anomaly_scores, relapse_labels, user_ids

    # ---------------------------------------------------
    # Compute metrics (AUROC, AUPRC)
    # ---------------------------------------------------
    def calculate_metrics(self, user: int, anomaly_scores, relapse_labels, user_ids, epoch_metrics):
        assert np.unique(user_ids) == user
        precision, recall, _ = sklearn.metrics.precision_recall_curve(relapse_labels, anomaly_scores)
        fpr, tpr, _ = sklearn.metrics.roc_curve(relapse_labels, anomaly_scores)
        auroc = sklearn.metrics.auc(fpr, tpr)
        auprc = sklearn.metrics.auc(recall, precision)
        return auroc, auprc

    # ---------------------------------------------------
    # Get training anomaly distribution (for normalization)
    # ---------------------------------------------------
    def get_train_dist_anomaly_scores(self, epoch: int, i: int):
        k = self.args.ensembles
        anomaly_scores = []
        for batch in self.dataloaders[i]["train_dist"]:
            if batch is None:
                continue
            x = batch["data"].to(self.args.device)
            features, _ = self.models[i](x)
            preds = self.mlps[i].forward(features[None, :, :].repeat([k, 1, 1]))
            average_pred = torch.mean(preds, 0)
            var_score = torch.sum((preds - average_pred) ** 2, dim=(2,))
            anomaly_score = torch.mean(torch.mean(var_score, 0)).item()
            anomaly_scores.append(anomaly_score)
        return anomaly_scores

    # ---------------------------------------------------
    # Validation Loop
    # ---------------------------------------------------
    def validate(self, epoch: int, epoch_metrics: dict, train_dist_anomaly_scores: dict):
        for i in range(len(self.models)):
            anomaly_scores, relapse_labels, user_ids = self.evaluate_ensembles(epoch, i, train_dist_anomaly_scores)
            auroc, auprc = self.calculate_metrics(i, anomaly_scores, relapse_labels, user_ids, epoch_metrics)
            avg = (auroc + auprc) / 2
            if avg > self.current_best_avgs[i]:
                self.current_best_avgs[i] = avg
                self.current_best_aurocs[i] = auroc
                self.current_best_auprcs[i] = auprc
                os.makedirs(self.args.save_path, exist_ok=True)
                if not os.path.exists(os.path.join(self.args.save_path, str(i+1))):
                    os.mkdir(f"{self.args.save_path}/{i+1}")
                torch.save(self.models[i].state_dict(), os.path.join(self.args.save_path, f"{i+1}/best_encoder.pth"))
                torch.save(self.mlps[i].state_dict(), os.path.join(self.args.save_path, f"{i+1}/best_ensembles.pth"))
                with open(os.path.join(self.args.save_path, f"{i+1}/train_dist_anomaly_scores.pkl"), "wb") as f:
                    pickle.dump(train_dist_anomaly_scores, f)

        for i in range(len(self.models)):
            print(f"P{str(i+1)} AUROC: {self.current_best_aurocs[i]:.4f}, "
                  f"AUPRC: {self.current_best_auprcs[i]:.4f}, "
                  f"AVG: {self.current_best_avgs[i]:.4f}")

        total_auroc = sum(self.current_best_aurocs) / len(self.models)
        total_auprc = sum(self.current_best_auprcs) / len(self.models)
        total_avg = (total_auroc + total_auprc) / 2

        print(f"TOTAL AUROC: {total_auroc:.4f}, AUPRC: {total_auprc:.4f}, "
              f"AVG: {total_avg:.4f}, Train Loss: {np.mean(epoch_metrics['loss_total']):.4f}")

    # ---------------------------------------------------
    # Full Training Loop
    # ---------------------------------------------------
    def train(self):
        for epoch in range(self.args.epochs):
            print("*" * 55)
            print(f"Start epoch {epoch+1}/{self.args.epochs}")
            epoch_metrics = {}
            train_dist_anomaly_scores = {}
            for i in range(len(self.models)):
                epoch_metrics = self.train_encoder_once(epoch, i, epoch_metrics)
                self.scheds[i].step()
                self.models[i].eval()
                self.train_ensemble(i)
                with torch.no_grad():
                    train_dist_anomaly_scores[i] = self.get_train_dist_anomaly_scores(epoch, i)

            with torch.no_grad():
                self.validate(epoch, epoch_metrics, train_dist_anomaly_scores)
