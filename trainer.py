import pickle
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sklearn.metrics
import os
from model import TransformerDualHead


# -------------------
# Circular loss helper
# -------------------
def circular_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute circular loss between predicted and true sin/cos pairs.
    Handles both (sin, cos) and multiple angles (e.g., sleep onset + wake).
    """
    if preds.shape[1] == 2:  # single angle (circadian)
        sin_pred, cos_pred = preds[:, 0], preds[:, 1]
        sin_true, cos_true = targets[:, 0], targets[:, 1]
        theta_pred = torch.atan2(sin_pred, cos_pred)
        theta_true = torch.atan2(sin_true, cos_true)
        return torch.mean(1 - torch.cos(theta_pred - theta_true))

    elif preds.shape[1] == 4:  # two angles (sleep onset, wake)
        loss = 0
        for i in range(0, 4, 2):
            sin_pred, cos_pred = preds[:, i], preds[:, i + 1]
            sin_true, cos_true = targets[:, i], targets[:, i + 1]
            theta_pred = torch.atan2(sin_pred, cos_pred)
            theta_true = torch.atan2(sin_true, cos_true)
            loss += torch.mean(1 - torch.cos(theta_pred - theta_true))
        return loss
    else:
        raise ValueError("Unexpected prediction shape for circular loss")


# -------------------
# Ensemble Head
# -------------------
class EnsembleLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int,
                 weight_decay: float = 0., bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.lin_w = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        if bias:
            self.lin_b = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (ensemble_size, batch, d_model)
        w_times_x = torch.bmm(x, self.lin_w)  # (ens, batch, out)
        y = torch.add(w_times_x, self.lin_b[:, None, :])
        return y

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lin_w, a=math.sqrt(5))
        if self.lin_b is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.lin_w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.lin_b, -bound, bound)


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


# -------------------
# Trainer Class
# -------------------
class Trainer:
    """ Class to train the dual-head transformer + ensembles """

    def __init__(self, models, optims, scheds, loaders, args):
        self.models = models
        self.optims = optims
        self.scheds = scheds
        self.dataloaders = loaders
        self.args = args

        self.current_best_avgs = [-np.inf for _ in range(len(self.models))]
        self.current_best_aurocs = [-np.inf for _ in range(len(self.models))]
        self.current_best_auprcs = [-np.inf for _ in range(len(self.models))]

        self.mlps = []
        self.optimizers = []
        for i in range(len(self.models)):
            ensemble_head = create_ensemble_mlp(self.args)
            self.mlps.append(ensemble_head)
            ps = ensemble_head.parameters()
            self.optimizers.append(torch.optim.Adam(ps, lr=1e-3, weight_decay=1e-4))

    # -------------------
    # Train encoder once
    # -------------------
    def train_encoder_once(self, epoch: int, i: int, epoch_metrics: dict):
        for batch in tqdm(self.dataloaders[i]['train'], desc=f'Train Encoder ({i+1}/{len(self.models)})'):
            if batch is None:
                continue

            x = batch['data'].to(self.args.device)  # (B, F, T)
            target_time = batch['target_time'].to(self.args.device)  # (B, 2)
            target_sleep = batch['target_sleep'].to(self.args.device)  # (B, 4)

            # Forward
            features, preds_time, preds_sleep = self.models[i](x)

            # Losses
            loss_time = circular_loss(preds_time, target_time)
            loss_sleep = circular_loss(preds_sleep, target_sleep)

            # weighted sum (λ can be tuned)
            loss = loss_time + self.args.sleep_loss_weight * loss_sleep

            self.optims[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models[i].parameters(), 5)
            self.optims[i].step()

            # log metrics
            metrics = {
                'loss_time': loss_time.item(),
                'loss_sleep': loss_sleep.item(),
                'loss_total': loss.item(),
            }
            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, []) + [v]

        return epoch_metrics

    # -------------------
    # Resample helper
    # -------------------
    def resample_batch(self, indices, dataset):
        ensemble_size = self.args.ensembles
        offsets = torch.zeros_like(indices)
        data_batch = []
        target_batch = []
        for i in range(indices.size()[0]):
            while (offsets[i] % ensemble_size) == 0:
                offsets[i] = torch.randint(low=1, high=len(dataset), size=(1,))
            r_idx = (offsets[i] + indices[i]) % len(dataset)
            x = dataset[r_idx]['data']
            t = dataset[r_idx]['target_time']  # ensembles only trained on circadian target
            data_batch.append(x)
            target_batch.append(t)
        data = torch.stack(data_batch).to(self.args.device)
        targets = torch.stack(target_batch).to(self.args.device)
        return data, targets

    # -------------------
    # Train ensemble
    # -------------------
    def train_ensemble(self, i: int):
        k = self.args.ensembles
        ensemble_head = self.mlps[i]
        ensemble_head.train()
        optim = self.optimizers[i]

        for batch in tqdm(self.dataloaders[i]['train'], desc=f'Train Ensemble ({i+1}/{len(self.models)})'):
            if batch is None:
                continue

            x = batch['data'].to(self.args.device)
            targets = batch['target_time'].to(self.args.device)
            features, _, _ = self.models[i](x)

            batched_features = features[None, :, :].repeat([k, 1, 1])
            batched_targets = targets[None, :, :].repeat([k, 1, 1])

            ensemble_mask = batch['idx'] % k
            resampled_x, r_targets = self.resample_batch(batch['idx'], self.dataloaders[i]['train'].dataset)
            r_features, _, _ = self.models[i](resampled_x)
            batched_features[ensemble_mask] = r_features
            batched_targets[ensemble_mask] = r_targets

            mean = ensemble_head.forward(batched_features)
            mse_loss = torch.sum(torch.pow(mean - batched_targets, 2), dim=(0, 2))
            total_loss = torch.mean(mse_loss)

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble_head.parameters(), 5)
            optim.step()

    # -------------------
    # Evaluate ensembles (robust target reshape)
    # -------------------
    def evaluate_ensembles(self, epoch: int, i: int, train_dist_anomaly_scores: dict):
        k = self.args.ensembles
        anomaly_scores = []
        relapse_labels = []
        user_ids = []

        _mean = np.mean(train_dist_anomaly_scores[i])
        _max, _min = np.max(train_dist_anomaly_scores[i]), np.min(train_dist_anomaly_scores[i])

        for batch in self.dataloaders[i]['val']:
            if batch is None:
                continue

            user_id = batch['user_id'].to(self.args.device)
            if user_id.item() >= self.args.num_patients:
                continue

            x = batch['data'].to(self.args.device)
            ensemble_head = self.mlps[user_id.item()]
            ensemble_head.eval()

            # Handle different input shapes for validation vs train
            if x.dim() == 4:  # validation mode: (1, num_windows, features, seq_len)
                x = x.squeeze(0)  # (num_windows, features, seq_len)
            
            # get circadian target - handle different shapes
            targets = batch['target_time'].to(self.args.device)
            
            # Handle different target shapes based on dataset mode
            if targets.dim() == 3:  # (1, num_windows, 2) in validation mode
                targets = targets.squeeze(0)  # (num_windows, 2)
            elif targets.dim() == 2:  # (batch_size, 2) in train mode
                pass  # already correct shape
            elif targets.dim() == 1:  # (2,) single target
                targets = targets.unsqueeze(0)  # (1, 2)
            
            N = targets.size(0)  # number of windows/timesteps
            
            # Ensure targets have correct dimensions for repeating
            if targets.dim() == 2:
                targets = targets.unsqueeze(0)  # (1, N, 2)
                targets = targets.repeat(k, 1, 1)   # (k, N, 2)
            else:
                # Fallback for unexpected shapes
                targets = targets.view(1, -1, 2).repeat(k, 1, 1)

            # Process each window through the model
            if x.dim() == 3:  # validation mode: (num_windows, features, seq_len)
                all_features = []
                for window_idx in range(x.size(0)):
                    window_x = x[window_idx:window_idx+1]  # (1, features, seq_len)
                    window_features, _, _ = self.models[i](window_x)
                    all_features.append(window_features)
                features = torch.cat(all_features, dim=0)  # (num_windows, d_model)
            else:
                # Train mode: (batch, features, seq_len)
                features, _, _ = self.models[i](x)
            
            # Handle features shape for ensemble
            if features.dim() == 2:  # (num_windows, d_model)
                batched_features = features.unsqueeze(0).repeat(k, 1, 1)  # (k, num_windows, d_model)
            else:
                batched_features = features.repeat(k, 1, 1)
                
            preds = ensemble_head.forward(batched_features)
            average_pred = torch.mean(preds, 0)

            var_score = torch.sum((preds - average_pred) ** 2, dim=(2,))
            mean_var = torch.mean(torch.mean(var_score, 0)).item()
            anomaly_score = (mean_var - _mean) / (_max - _min)

            anomaly_scores.append(anomaly_score)
            relapse_labels.append(batch['relapse_label'].item())
            user_ids.append(batch['user_id'].item())

        anomaly_scores = (np.array(anomaly_scores) > 0.0).astype(np.float64)
        relapse_labels = np.array(relapse_labels)
        user_ids = np.array(user_ids)
        return anomaly_scores, relapse_labels, user_ids 
    # -------------------
    # Metrics
    # -------------------
    def calculate_metrics(self, user: int, anomaly_scores, relapse_labels, user_ids, epoch_metrics):
        assert np.unique(user_ids) == user
        precision, recall, _ = sklearn.metrics.precision_recall_curve(relapse_labels, anomaly_scores)
        fpr, tpr, _ = sklearn.metrics.roc_curve(relapse_labels, anomaly_scores)
        auroc = sklearn.metrics.auc(fpr, tpr)
        auprc = sklearn.metrics.auc(recall, precision)
        return auroc, auprc

    # -------------------
    # Train dist anomaly scores
    # -------------------
    def get_train_dist_anomaly_scores(self, epoch: int, i: int):
        k = self.args.ensembles
        anomaly_scores = []
        for batch in self.dataloaders[i]['train_dist']:
            if batch is None:
                continue
            x = batch['data'].to(self.args.device)
            ensemble_head = self.mlps[i]
            features, _, _ = self.models[i](x)
            batched_features = features[None, :, :].repeat([k, 1, 1])
            preds = ensemble_head.forward(batched_features)
            average_pred = torch.mean(preds, 0)
            var_score = torch.sum((preds - average_pred) ** 2, dim=(2,))
            anomaly_score = torch.mean(torch.mean(var_score, 0)).item()
            anomaly_scores.append(anomaly_score)
        return anomaly_scores

    # -------------------
    # Validation
    # -------------------
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
                    os.mkdir(f'{self.args.save_path}/{i+1}')
                torch.save(self.models[i].state_dict(),
                           os.path.join(self.args.save_path, f'{i+1}/best_encoder.pth'))
                torch.save(self.mlps[i].state_dict(),
                           os.path.join(self.args.save_path, f'{i+1}/best_ensembles.pth'))
                train_dist_path = os.path.join(self.args.save_path, f'{i+1}/train_dist_anomaly_scores.pkl')
                with open(train_dist_path, 'wb') as f:
                    pickle.dump(train_dist_anomaly_scores, f)

        for i in range(len(self.models)):
            print(f"P{str(i + 1)} AUROC: {self.current_best_aurocs[i]:.4f}, "
                  f"AUPRC: {self.current_best_auprcs[i]:.4f}, "
                  f"AVG: {self.current_best_avgs[i]:.4f}")

        total_auroc = sum(self.current_best_aurocs) / len(self.models)
        total_auprc = sum(self.current_best_auprcs) / len(self.models)
        total_avg = (total_auroc + total_auprc) / 2
        print(f'TOTAL\tAUROC: {total_auroc:.4f},  AUPRC: {total_auprc:.4f}, Total AVG: {total_avg:.4f}, '
              f"Train Loss: {np.mean(epoch_metrics['loss_total']):.4f}")

    # -------------------
    # Training loop
    # -------------------
    def train(self):
        for epoch in range(self.args.epochs):
            print("*" * 55)
            print(f"Start epoch {epoch+1}/{self.args.epochs}")
            epoch_metrics = {}
            train_dist_anomaly_scores = {}

            for i in range(len(self.models)):
                self.models[i].train()
                epoch_metrics = self.train_encoder_once(epoch, i, epoch_metrics)
                self.scheds[i].step()

                self.models[i].eval()
                self.train_ensemble(i)
                with torch.no_grad():
                    train_dist_anomaly_scores[i] = self.get_train_dist_anomaly_scores(epoch, i)

            with torch.no_grad():
                self.validate(epoch, epoch_metrics, train_dist_anomaly_scores)
