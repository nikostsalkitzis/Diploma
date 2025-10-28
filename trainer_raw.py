import pickle
import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sklearn.metrics
from model_raw import EnsembleLinear


def create_ensemble_mlp(args):
    """
    Builds the per-patient ensemble head MLP.
    Uses EnsembleLinear layers so we get k parallel predictors.
    """
    m = nn.Sequential(
        EnsembleLinear(args.d_model, args.d_model, args.ensembles),
        nn.ReLU(),
        EnsembleLinear(args.d_model, args.d_model, args.ensembles),
        nn.ReLU(),
        EnsembleLinear(args.d_model, args.output_dim, args.ensembles),
    )
    m.to(args.device)
    return m


class Trainer:
    """
    Trainer for:
      - TransformerHeartPredictor encoder
      - Per-patient ensemble MLP heads
      - Uncertainty-based relapse scoring

    Accelerated with:
      - CUDA non_blocking transfers
      - AMP autocast + GradScaler
      - persistent_workers / pinned memory handled in train_raw.py
    """

    def __init__(self, models, optims, scheds, loaders, args):
        self.models = models
        self.optims = optims
        self.scheds = scheds
        self.dataloaders = loaders
        self.args = args

        # Standard regression loss
        self.regression_loss = nn.MSELoss()

        # Track best validation metrics so we can checkpoint
        self.current_best_avgs = [-np.inf for _ in range(len(self.models))]
        self.current_best_aurocs = [-np.inf for _ in range(len(self.models))]
        self.current_best_auprcs = [-np.inf for _ in range(len(self.models))]

        # Per-patient ensemble heads + optimizers
        self.mlps = []
        self.optimizers = []

        # AMP scalers
        self.scalers = []             # for encoder optimizers
        self.ensemble_scalers = []    # for ensemble optimizers

        for _ in range(len(self.models)):
            ensemble_head = create_ensemble_mlp(self.args)
            self.mlps.append(ensemble_head)
            ps = ensemble_head.parameters()
            self.optimizers.append(
                torch.optim.Adam(ps, lr=1e-3, weight_decay=1e-4)
            )

        # 1 scaler per encoder optimizer
        for _ in range(len(self.models)):
            self.scalers.append(torch.cuda.amp.GradScaler(enabled=("cuda" in self.args.device)))
            self.ensemble_scalers.append(torch.cuda.amp.GradScaler(enabled=("cuda" in self.args.device)))

    # ---------------------------------------------------
    # Stage 1: Train the Transformer encoder (heart regression)
    # ---------------------------------------------------
    def train_encoder_once(self, epoch: int, i: int, epoch_metrics: dict):
        """
        Train encoder model i to predict heart/HRV metrics from raw windows.
        Uses AMP for speed on GPU.
        """
        device = self.args.device
        model = self.models[i]
        model.train()

        scaler = self.scalers[i]
        optim = self.optims[i]

        pbar = tqdm(self.dataloaders[i]["train"], desc=f"Train Encoder ({i+1}/{len(self.models)})", leave=False)

        for batch in pbar:
            if batch is None:
                continue

            # Move batch to GPU (non_blocking helps with pinned memory DataLoader)
            x = batch["data"].to(device, non_blocking=True)          # (B, C, T)
            heart_labels = batch["target"].to(device, non_blocking=True)  # (B, 5, 1)
            heart_labels = torch.squeeze(heart_labels, dim=-1)            # (B, 5)

            optim.zero_grad(set_to_none=True)

            # autocast for mixed precision forward+loss
            with torch.cuda.amp.autocast(enabled=("cuda" in device)):
                features, heart_preds = model(x)  # features:(B,d_model), heart_preds:(B,5)
                loss_heart = self.regression_loss(heart_preds, heart_labels)
                total_loss = loss_heart

            # backward with GradScaler
            scaler.scale(total_loss).backward()

            # clip grads before step, but need to unscale first
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            scaler.step(optim)
            scaler.update()

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
        """
        Bootstrapping trick to make different ensemble members see
        slightly different target mappings, encouraging diversity.
        """
        ensemble_size = self.args.ensembles
        offsets = torch.zeros_like(indices)
        data_batch = []
        target_batch = []

        # loop in python; typically batch is small so it's not terrible
        for i in range(indices.size()[0]):
            while (offsets[i] % ensemble_size) == 0:
                offsets[i] = torch.randint(low=1, high=len(dataset), size=(1,), device=indices.device)
            r_idx = (offsets[i] + indices[i]) % len(dataset)

            x = dataset[int(r_idx.item())]["data"]
            t = dataset[int(r_idx.item())]["target"]

            data_batch.append(x)
            target_batch.append(t)

        data = torch.stack(data_batch)
        targets = torch.stack(target_batch)
        return data, targets

    # ---------------------------------------------------
    # Stage 2: Train ensemble head (uncertainty model)
    # ---------------------------------------------------
    def train_ensemble(self, i: int):
        """
        Train ensemble MLP on top of frozen encoder features to
        predict heart targets. High disagreement => anomaly.

        Uses AMP.
        """
        device = self.args.device
        k = self.args.ensembles

        encoder = self.models[i]
        encoder.eval()  # freeze encoder during head training

        ensemble_head = self.mlps[i]
        ensemble_head.train()

        optim = self.optimizers[i]
        scaler = self.ensemble_scalers[i]

        pbar = tqdm(self.dataloaders[i]["train"], desc=f"Train Ensemble ({i+1}/{len(self.models)})", leave=False)

        for batch in pbar:
            if batch is None:
                continue

            # batch data
            x = batch["data"].to(device, non_blocking=True)               # (B, C, T)
            targets = batch["target"].to(device, non_blocking=True)       # (B, 5, 1)
            targets = torch.squeeze(targets, dim=-1)                       # (B, 5)
            idxs = batch["idx"].to(device, non_blocking=True).long()       # (B,)

            optim.zero_grad(set_to_none=True)

            # (1) Encode current batch
            with torch.cuda.amp.autocast(enabled=("cuda" in device)):
                features, _ = encoder(x)   # features: (B, d_model)

            # (2) Encode a "resampled" batch to diversify ensemble members
            resampled_x, r_targets = self.resample_batch(idxs, self.dataloaders[i]["train"].dataset)
            resampled_x = resampled_x.to(device, non_blocking=True)
            r_targets = r_targets.to(device, non_blocking=True)  # (B, 5, 1)
            r_targets = torch.squeeze(r_targets, dim=-1)         # (B, 5)

            with torch.cuda.amp.autocast(enabled=("cuda" in device)):
                r_features, _ = encoder(resampled_x)  # (B, d_model)

                # Prepare ensemble inputs
                # batched_features: (k, B, d_model)
                # batched_targets:  (k, B, 5)
                batched_features = features[None, :, :].repeat([k, 1, 1])
                batched_targets = targets[None, :, :].repeat([k, 1, 1])

                # each sample chooses which ensemble member gets the resampled version
                ensemble_mask = idxs % k  # (B,)

                # replace that ensemble member's row with resampled (diversity trick)
                for bb in range(batched_features.size(1)):
                    m_id = ensemble_mask[bb]
                    batched_features[m_id, bb, :] = r_features[bb, :]
                    batched_targets[m_id, bb, :]  = r_targets[bb, :]

                # forward through ensemble head
                mean_pred = ensemble_head.forward(batched_features)  # (k, B, 5)

                # mse loss across ensemble members, summed, averaged over batch
                mse_loss = torch.sum(torch.pow(mean_pred - batched_targets, 2), dim=(0, 2))  # (B,)
                total_loss = torch.mean(mse_loss)

            scaler.scale(total_loss).backward()

            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(ensemble_head.parameters(), 5.0)

            scaler.step(optim)
            scaler.update()

    # ---------------------------------------------------
    # Train-dist anomaly reference distribution
    # ---------------------------------------------------
    def get_train_dist_anomaly_scores(self, epoch: int, i: int):
        """
        Get baseline variance distribution for patient i on their train_dist set.
        We'll use this to normalize anomaly scores at eval time.
        """
        device = self.args.device
        k = self.args.ensembles

        encoder = self.models[i]
        encoder.eval()

        ensemble_head = self.mlps[i]
        ensemble_head.eval()

        anomaly_scores = []

        with torch.no_grad():
            for batch in self.dataloaders[i]["train_dist"]:
                if batch is None:
                    continue

                x = batch["data"].to(device, non_blocking=True)  # (B, C, T)
                features, _ = encoder(x)                         # (B, d_model)

                # repeat for ensemble members
                bf = features[None, :, :].repeat([k, 1, 1])      # (k,B,d_model)
                preds = ensemble_head(bf)                        # (k,B,5)

                avg = torch.mean(preds, 0)                       # (B,5)
                var_score = torch.sum((preds - avg) ** 2, dim=2) # (k,B)
                # mean over ensemble members and batch
                anomaly_score = torch.mean(torch.mean(var_score, 0)).item()
                anomaly_scores.append(anomaly_score)

        return anomaly_scores

    # ---------------------------------------------------
    # Validation helper: compute anomaly scores per day
    # ---------------------------------------------------
    def evaluate_ensembles(self, epoch: int, i: int, train_dist_anomaly_scores: dict):
        """
        For each validation "day" item:
          - run windows through encoder + ensemble
          - compute ensemble variance as anomaly score
          - normalize wrt the train_dist reference
          - map to binary 0/1 with >0 threshold
        """
        device = self.args.device
        k = self.args.ensembles

        encoder = self.models[i]
        ensemble_head = self.mlps[i]

        encoder.eval()
        ensemble_head.eval()

        # stats from train_dist for normalization
        dist = np.array(train_dist_anomaly_scores[i])
        _mean = np.mean(dist)
        _max = np.max(dist)
        _min = np.min(dist)
        denom = (_max - _min) if (_max - _min) != 0 else 1.0

        anomaly_scores = []
        relapse_labels = []
        user_ids = []

        with torch.no_grad():
            for batch in self.dataloaders[i]["val"]:
                if batch is None:
                    continue

                user_id = batch["user_id"].to(device, non_blocking=True)
                if user_id.item() >= self.args.num_patients:
                    continue

                # shape from dataset val: (1, M, C, T) or (M, C, T)?
                # We built dataset so "data" for val is shape (M,C,T) already,
                # but DataLoader(batch_size=1) wraps it to (1,M,C,T).
                x = batch["data"].squeeze(0).to(device, non_blocking=True)  # (M,C,T)
                targets = batch["target"].squeeze(0).to(device, non_blocking=True)  # (M,5)

                # run encoder on all windows at once
                feats, _ = encoder(x)  # feats: (M,d_model)

                # ensemble predictions for each window
                bf = feats[None, :, :].repeat([k, 1, 1])  # (k,M,d_model)
                preds = ensemble_head(bf)                 # (k,M,5)

                avg = torch.mean(preds, 0)                # (M,5)
                var_score = torch.sum((preds - avg) ** 2, dim=2)  # (k,M)
                mean_var = torch.mean(torch.mean(var_score, 0)).item()

                # normalize
                anomaly_score = (mean_var - _mean) / denom

                # convert to binary anomaly decision for metrics
                anomaly_scores.append(1.0 if anomaly_score > 0 else 0.0)
                relapse_labels.append(batch["relapse_label"].item())
                user_ids.append(batch["user_id"].item())

        anomaly_scores = np.array(anomaly_scores, dtype=np.float64)
        relapse_labels = np.array(relapse_labels, dtype=np.int64)
        user_ids = np.array(user_ids, dtype=np.int64)

        return anomaly_scores, relapse_labels, user_ids

    # ---------------------------------------------------
    # Metrics
    # ---------------------------------------------------
    def calculate_metrics(self, user: int, anomaly_scores, relapse_labels, user_ids, epoch_metrics):
        """
        Computes AUROC and AUPRC for a given user.
        relapse_labels: 0/1 ground truth
        anomaly_scores: 0/1 predicted anomaly (after threshold)
        """
        assert np.unique(user_ids) == user
        precision, recall, _ = sklearn.metrics.precision_recall_curve(relapse_labels, anomaly_scores)
        fpr, tpr, _ = sklearn.metrics.roc_curve(relapse_labels, anomaly_scores)
        auroc = sklearn.metrics.auc(fpr, tpr)
        auprc = sklearn.metrics.auc(recall, precision)
        return auroc, auprc

    # ---------------------------------------------------
    # Validation loop
    # ---------------------------------------------------
    def validate(self, epoch: int, epoch_metrics: dict, train_dist_anomaly_scores: dict):
        """
        Runs validation across all patients, updates best metrics,
        and saves checkpoints if we improved.
        """
        for i in range(len(self.models)):
            anomaly_scores, relapse_labels, user_ids = self.evaluate_ensembles(epoch, i, train_dist_anomaly_scores)
            if len(np.unique(relapse_labels)) < 2:
                # cannot compute AUROC/AUPRC with only one class in labels
                continue

            auroc, auprc = self.calculate_metrics(i, anomaly_scores, relapse_labels, user_ids, epoch_metrics)
            avg = (auroc + auprc) / 2.0

            if avg > self.current_best_avgs[i]:
                self.current_best_avgs[i] = avg
                self.current_best_aurocs[i] = auroc
                self.current_best_auprcs[i] = auprc

                os.makedirs(self.args.save_path, exist_ok=True)
                patient_ckpt_dir = os.path.join(self.args.save_path, str(i + 1))
                if not os.path.exists(patient_ckpt_dir):
                    os.mkdir(patient_ckpt_dir)

                torch.save(self.models[i].state_dict(), os.path.join(patient_ckpt_dir, "best_encoder.pth"))
                torch.save(self.mlps[i].state_dict(), os.path.join(patient_ckpt_dir, "best_ensembles.pth"))
                with open(os.path.join(patient_ckpt_dir, "train_dist_anomaly_scores.pkl"), "wb") as f:
                    pickle.dump(train_dist_anomaly_scores, f)

        # after updating all patients, report
        for i in range(len(self.models)):
            print(
                f"P{str(i+1)} AUROC: {self.current_best_aurocs[i]:.4f}, "
                f"AUPRC: {self.current_best_auprcs[i]:.4f}, "
                f"AVG: {self.current_best_avgs[i]:.4f}"
            )

        # macro average (only among patients we've updated)
        valid_avgs = [a for a in self.current_best_avgs if a != -np.inf]
        valid_aurocs = [a for a in self.current_best_aurocs if a != -np.inf]
        valid_auprcs = [a for a in self.current_best_auprcs if a != -np.inf]

        if valid_avgs:
            total_auroc = float(np.mean(valid_aurocs))
            total_auprc = float(np.mean(valid_auprcs))
            total_avg = (total_auroc + total_auprc) / 2.0
        else:
            total_auroc = 0.0
            total_auprc = 0.0
            total_avg = 0.0

        mean_train_loss = np.mean(epoch_metrics["loss_total"]) if "loss_total" in epoch_metrics else 0.0

        print(
            f"TOTAL AUROC: {total_auroc:.4f}, "
            f"AUPRC: {total_auprc:.4f}, "
            f"AVG: {total_avg:.4f}, "
            f"Train Loss: {mean_train_loss:.4f}"
        )

    # ---------------------------------------------------
    # Full training loop
    # ---------------------------------------------------
    def train(self):
        """
        Full multi-epoch training:
          1. train encoder
          2. step LR
          3. train ensemble head
          4. build train_dist_anomaly_scores
          5. validate + save best checkpoints
        """
        for epoch in range(self.args.epochs):
            print("*" * 55)
            print(f"Start epoch {epoch+1}/{self.args.epochs}")

            epoch_metrics = {}
            train_dist_anomaly_scores = {}

            for i in range(len(self.models)):
                # Stage A: train encoder i
                epoch_metrics = self.train_encoder_once(epoch, i, epoch_metrics)

                # LR schedule step for this encoder
                self.scheds[i].step()

                # Stage B: train ensemble head for patient i
                # (encoder put into eval internally)
                self.train_ensemble(i)

                # Stage C: build train-dist baseline for normalization
                with torch.no_grad():
                    train_dist_anomaly_scores[i] = self.get_train_dist_anomaly_scores(epoch, i)

            # Validation pass
            with torch.no_grad():
                self.validate(epoch, epoch_metrics, train_dist_anomaly_scores)
