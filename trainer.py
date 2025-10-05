import pickle
import math
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import sklearn.metrics
import os


class EnsembleLinear(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, ensemble_size: int,
                 weight_decay: float = 0., bias: bool = True) -> None:
        super(EnsembleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.lin_w = nn.Parameter(torch.Tensor(ensemble_size, in_features, out_features))
        self.weight_decay = weight_decay
        if bias:
            self.lin_b = nn.Parameter(torch.Tensor(ensemble_size, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_times_x = torch.bmm(x, self.lin_w)
        y = torch.add(w_times_x, self.lin_b[:, None, :])  # w times x + b
        return y

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.lin_b is not None
        )

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
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


class Trainer:
    ''' Class to train the multi-task classifier '''

    def __init__(self, models, optims, scheds, loaders, args):

        self.models = models
        self.optims = optims
        self.scheds = scheds
        self.dataloaders = loaders
        self.args = args
        
        # Multi-task loss functions
        self.time_criterion = nn.MSELoss()  # For time prediction (regression)
        self.activity_criterion = nn.CrossEntropyLoss()  # For activity classification
        
        # Loss weights (adjust based on importance)
        self.time_loss_weight = getattr(args, 'time_loss_weight', 1.0)
        self.activity_loss_weight = getattr(args, 'activity_loss_weight', 0.5)

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

    def train_encoder_once(self, epoch: int, i: int, epoch_metrics: dict):
        for batch in tqdm(self.dataloaders[i]['train'], desc=f'Train Encoder ({i+1}/{len(self.models)})'):

            if batch is None:
                continue

            # x has shape: (16, 6, 24) == (batch_size, input_features, seq_len)
            x = batch['data'].to(self.args.device)
            time_labels = batch['time_target'].to(self.args.device)  # Time prediction labels
            activity_labels = batch['activity_target'].to(self.args.device)  # Activity classification labels

            # Forward pass - multi-task
            outputs = self.models[i](x)
            time_pred = outputs['time_pred']
            activity_pred = outputs['activity_pred']

            # Calculate both losses
            time_loss = self.time_criterion(time_pred, time_labels)
            activity_loss = self.activity_criterion(activity_pred, activity_labels)
            
            # Combined loss
            total_loss = (self.time_loss_weight * time_loss + 
                         self.activity_loss_weight * activity_loss)

            self.optims[i].zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models[i].parameters(), 5)

            self.optims[i].step()

            # Log metrics
            metrics = {
                'total_loss': total_loss.item(),
                'time_loss': time_loss.item(),
                'activity_loss': activity_loss.item(),
            }

            for k, v in metrics.items():
                epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]

        return epoch_metrics

    def resample_batch(self, indices, dataset):
        ensemble_size = self.args.ensembles
        offsets = torch.zeros_like(indices)
        data_batch = []
        time_target_batch = []
        activity_target_batch = []
        for i in range(indices.size()[0]):
            while (offsets[i] % ensemble_size) == 0:
                offsets[i] = torch.randint(low=1, high=len(dataset), size=(1,))
            r_idx = (offsets[i] + indices[i]) % len(dataset)
            x = dataset[r_idx]['data']
            t_time = dataset[r_idx]['time_target']
            t_activity = dataset[r_idx]['activity_target']
            data_batch.append(x)
            time_target_batch.append(t_time)
            activity_target_batch.append(t_activity)
        data = torch.stack(data_batch).to(self.args.device)
        time_targets = torch.stack(time_target_batch).to(self.args.device)
        activity_targets = torch.stack(activity_target_batch).to(self.args.device)
        return data, time_targets, activity_targets

    def train_ensemble(self, i: int):
        k = self.args.ensembles
        ensemble_head = self.mlps[i]
        ensemble_head.train()
        optim = self.optimizers[i]
        size = len(self.dataloaders[i]['train'])

        for batch in tqdm(self.dataloaders[i]['train'], desc=f'Train Ensemble ({i+1}/{len(self.models)})'):
            if batch is None:
                continue

            # x has shape: (16, 6, 24) == (batch_size, input_features, seq_len)
            x = batch['data'].to(self.args.device)
            time_targets = batch['time_target'].to(self.args.device)
            # activity_targets = batch['activity_target'].to(self.args.device)  # Not used in ensemble training

            # Forward - only use time prediction for ensemble training (main task)
            outputs = self.models[i](x)
            features = outputs['features']

            batched_features = features[None, :, :].repeat([k, 1, 1])
            batched_targets = time_targets[None, :, :].repeat([k, 1, 1])
            
            ensemble_mask = batch['idx'] % k

            resampled_x, r_time_targets, _ = self.resample_batch(
                batch['idx'],
                self.dataloaders[i]['train'].dataset
            )
            
            # Forward pass for resampled data
            r_outputs = self.models[i](resampled_x)
            r_features = r_outputs['features']
            
            batched_features[ensemble_mask] = r_features
            batched_targets[ensemble_mask] = r_time_targets

            # forward pass - ensemble only for time prediction
            mean = ensemble_head.forward(batched_features)
            mse_loss = torch.sum(torch.pow(mean - batched_targets, 2), dim=(0, 2))
            total_loss = torch.mean(mse_loss)

            optim.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ensemble_head.parameters(), 5)

            optim.step()

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

            # .Note: using batch_size = 1  -- x.shape = (1, 16, 6, 24)
            x = batch['data'].to(self.args.device)
            x = x.squeeze(0)  # remove fake batch dimension
            ensemble_head = self.mlps[user_id.item()]
            ensemble_head.eval()
            
            # Only use time targets for ensemble evaluation
            time_targets = batch['time_target'].to(self.args.device)
            time_targets = torch.squeeze(time_targets)
            if time_targets.ndim < 2:
                time_targets = torch.reshape(time_targets, (1, -1))

            # targets.shape: (ens_size, num_day_samples, out_dim)
            time_targets = time_targets[None, :, :].repeat([k, 1, 1])

            # Get features from model
            outputs = self.models[i](x)
            features = outputs['features']

            batched_features = features[None, :, :].repeat([k, 1, 1])
            
            # preds.shape = (ens_size, num_day_samples, output_dim)
            preds = ensemble_head.forward(batched_features)
            average_pred = torch.mean(preds, 0)

            var_score = torch.sum((preds - average_pred)**2, dim=(2, ))
            mean_var = torch.mean(torch.mean(var_score, 0)).item()
            anomaly_score = (mean_var - _mean) / (_max - _min)

            anomaly_scores.append(anomaly_score)
            relapse_labels.append(batch['relapse_label'].item())
            user_ids.append(batch['user_id'].item())

        anomaly_scores = (np.array(anomaly_scores) > 0.0).astype(np.float64)
        relapse_labels = np.array(relapse_labels)
        user_ids = np.array(user_ids)
        return anomaly_scores, relapse_labels, user_ids

    def calculate_metrics(self, user: int, anomaly_scores, relapse_labels, user_ids, epoch_metrics):

        assert np.unique(user_ids) == user

        # Compute ROC Curve
        precision, recall, _ = sklearn.metrics.precision_recall_curve(relapse_labels, anomaly_scores)

        fpr, tpr, _ = sklearn.metrics.roc_curve(relapse_labels, anomaly_scores)

        # # Compute AUROC
        auroc = sklearn.metrics.auc(fpr, tpr)

        # # Compute AUPRC
        auprc = sklearn.metrics.auc(recall, precision)

        avg = (auroc + auprc) / 2
        # print(f'\tUSER: {user}, AUROC: {auroc:.4f}, AUPRC: {auprc:.4f}, AVG: {avg:.4f}')
        return auroc, auprc

    def get_train_dist_anomaly_scores(self, epoch: int, i: int):
        k = self.args.ensembles
        anomaly_scores = []
        for batch in self.dataloaders[i]['train_dist']:

            if batch is None:
                continue

            x = batch['data'].to(self.args.device)
            ensemble_head = self.mlps[i]
            
            # Get features from model
            outputs = self.models[i](x)
            features = outputs['features']

            batched_features = features[None, :, :].repeat([k, 1, 1])
            # preds.shape = (ens_size, num_day_samples, output_dim)
            preds = ensemble_head.forward(batched_features)
            average_pred = torch.mean(preds, 0)

            var_score = torch.sum((preds - average_pred)**2, dim=(2, ))
            anomaly_score = torch.mean(torch.mean(var_score, 0)).item()
            anomaly_scores.append(anomaly_score)
        return anomaly_scores

    def validate(self, epoch: int, epoch_metrics: dict, train_dist_anomaly_scores: dict):
        for i in range(len(self.models)):
            # print('Calculating accuracy on validation set and anomaly scores...')
            anomaly_scores, relapse_labels, user_ids = self.evaluate_ensembles(epoch, i, train_dist_anomaly_scores)

            # print('Calculating metrics...')
            auroc, auprc = self.calculate_metrics(i, anomaly_scores, relapse_labels, user_ids, epoch_metrics)

            # save best model
            avg = (auroc + auprc) / 2
            if avg > self.current_best_avgs[i]:
                self.current_best_avgs[i] = avg
                self.current_best_aurocs[i] = auroc
                self.current_best_auprcs[i] = auprc
                os.makedirs(self.args.save_path, exist_ok=True)
                if not os.path.exists(os.path.join(self.args.save_path, str(i+1))):
                    os.makedirs(f'{self.args.save_path}/{i+1}', exist_ok=True)
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

        print(f'MULTI-TASK TOTAL\tAUROC: {total_auroc:.4f},  AUPRC: {total_auprc:.4f}, Total AVG: {total_avg:.4f}, '
              f'Total Loss: {np.mean(epoch_metrics["total_loss"]):.4f}, '
              f'Time Loss: {np.mean(epoch_metrics["time_loss"]):.4f}, '
              f'Activity Loss: {np.mean(epoch_metrics["activity_loss"]):.4f}')


    def train(self):
        for epoch in range(self.args.epochs):
            print("*"*55)
            print(f"Start epoch {epoch+1}/{self.args.epochs} - Multi-Task Learning")
            epoch_metrics = {}
            train_dist_anomaly_scores = {}
            for i in range(len(self.models)):

                # ------ start training ------ #
                self.models[i].train()
                epoch_metrics = self.train_encoder_once(epoch, i, epoch_metrics)
                self.scheds[i].step()

                self.models[i].eval()
                self.train_ensemble(i)
                with torch.no_grad():
                    train_dist_anomaly_scores[i] = self.get_train_dist_anomaly_scores(epoch, i)

            # ------ start validating ------ #
            with torch.no_grad():
                self.validate(epoch, epoch_metrics, train_dist_anomaly_scores)
