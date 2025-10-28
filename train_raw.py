import os
import pickle
import torch
import argparse
from torch.optim.lr_scheduler import MultiStepLR

from dataset_raw import PatientDataset
from model_raw import TransformerHeartPredictor
from trainer_raw import Trainer


def get_device(device_str="auto") -> str:
    """
    Picks device to run on.
    device_str can be:
      - "auto"  -> choose cuda if available else cpu
      - "cuda", "cuda:0", "cpu", etc.
    """
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    else:
        # validate
        try:
            torch.device(device_str)
            return device_str
        except Exception as e:
            print("Invalid --device, falling back to CPU. Error:", e)
            return "cpu"


def parse_args():
    p = argparse.ArgumentParser(
        description="Train raw-signal Transformer + ensemble relapse detector"
    )

    # Compute / system
    p.add_argument("--cores", type=int, default=os.cpu_count())
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--ensembles", type=int, default=5)

    # Patients
    p.add_argument("--num_patients", type=int, default=8)

    # Data / paths
    p.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Root path with raw data. Expect P1/, P2/, ... each containing train_*, val_*, test_* folders",
    )
    p.add_argument(
        "--save_path",
        type=str,
        default="checkpoints_raw",
        help="Where to save per-patient checkpoints and scalers",
    )

    # Windowing hyperparams (in 5-minute bins)
    p.add_argument("--window_size", type=int, default=24, help="How many 5-min bins per window (24 -> 2 hours)")
    p.add_argument("--stride", type=int, default=12, help="How many 5-min bins we slide between windows (12 -> 1 hour stride)")

    # Model hyperparams
    p.add_argument("--input_features", type=int, default=10, help="10 raw channels (acc,gyr,HR,RR,sin,cos)")
    p.add_argument("--output_dim", type=int, default=5, help="We predict 5 heart/HRV summary metrics")
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--dim_feedforward_encoder", type=int, default=2048)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--nlayers", type=int, default=2)
    p.add_argument("--patch_stride", type=int, default=25,
                   help="Conv1d stride for patch embedding (downsamples long sequence before Transformer)")

    # Optimization
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=50)

    args = p.parse_args()
    args.seq_len = args.window_size  # legacy compat (not super-used now)

    # resolve device now
    args.device = get_device(args.device)

    return args


def main():
    args = parse_args()

    print(f"🧠 Using device: {args.device}")

    # ---------------------------------------------------
    # 1. Build one model per patient
    # ---------------------------------------------------
    # Each patient gets:
    # - their own TransformerHeartPredictor
    # - their own optimizer and scheduler
    # - their own dataset splits
    #
    # This is consistent with your original design where every Pk had its own model head.
    #
    models = []
    for _ in range(args.num_patients):
        model = TransformerHeartPredictor(vars(args)).to(args.device)
        models.append(model)

    n_parameters = sum(p.numel() for p in models[0].parameters() if p.requires_grad)
    print("Number of encoder parameters:", n_parameters)

    # ---------------------------------------------------
    # 2. Create optimizers & schedulers for each patient model
    # ---------------------------------------------------
    optimizers = []
    schedulers = []

    for i in range(args.num_patients):
        opt = torch.optim.Adam(
            params=models[i].parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay,
        )
        optimizers.append(opt)

        sched = MultiStepLR(
            opt,
            milestones=[args.epochs // 2, (args.epochs * 3) // 4],
            gamma=0.1,
        )
        schedulers.append(sched)

    # ---------------------------------------------------
    # 3. Build datasets and scalers per patient
    # ---------------------------------------------------
    # Important:
    #   We create three datasets per patient:
    #     - train:            overlapping windows from train_* folders (relapse=0)
    #     - val:              full-day items from val_* folders
    #     - train_dist:       basically same as train (used to build baseline variance distribution)
    #
    #   We ALSO save a per-patient scaler (fit on that patient's train windows),
    #   just like your original code did.
    #
    train_datasets = []
    train_dist_datasets = []
    valid_datasets = []

    os.makedirs(args.save_path, exist_ok=True)

    for patient in [f"P{i}" for i in range(1, args.num_patients + 1)]:
        # 3a. TRAIN dataset for this patient
        train_dataset = PatientDataset(
            dataset_path=args.dataset_path,
            patient=patient,
            mode="train",
            window_size=args.window_size,
            stride=args.stride,
            scaler=None,  # fit scaler here
        )
        train_datasets.append(train_dataset)

        # 3b. Save the fitted scaler for this patient
        patient_id = patient[1:]  # "P3" -> "3"
        patient_ckpt_dir = os.path.join(args.save_path, patient_id)
        os.makedirs(patient_ckpt_dir, exist_ok=True)

        with open(os.path.join(patient_ckpt_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(train_dataset.scaler, f)

        # 3c. VAL dataset (evaluation / relapse scoring)
        valid_dataset = PatientDataset(
            dataset_path=args.dataset_path,
            patient=patient,
            mode="val",
            window_size=args.window_size,
            stride=args.stride,
            scaler=train_dataset.scaler,  # reuse train scaler
        )
        valid_datasets.append(valid_dataset)

        # 3d. TRAIN_DIST dataset
        # This is how we compute the "normal baseline" distribution of ensemble variance
        train_dist_dataset = PatientDataset(
            dataset_path=args.dataset_path,
            patient=patient,
            mode="train",
            window_size=args.window_size,
            stride=args.stride,
            scaler=train_dataset.scaler,  # use same scaler
        )
        train_dist_datasets.append(train_dist_dataset)

    # ---------------------------------------------------
    # 4. Wrap datasets in DataLoaders
    # ---------------------------------------------------
    # We use:
    #   pin_memory=True            -> faster CPU->GPU transfer
    #   persistent_workers=True    -> workers don't die after each epoch
    #   prefetch_factor=2 or 4     -> overlap data prep and training
    #
    # Note: persistent_workers requires num_workers > 0
    #
    all_loaders = []
    for i in range(args.num_patients):
        loaders = {
            "train": torch.utils.data.DataLoader(
                train_datasets[i],
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cores,
                pin_memory=True,
                persistent_workers=True if args.cores and args.cores > 0 else False,
                prefetch_factor=2 if args.cores and args.cores > 0 else None,
                drop_last=False,
            ),
            "val": torch.utils.data.DataLoader(
                valid_datasets[i],
                batch_size=1,
                shuffle=False,
                num_workers=args.cores,
                pin_memory=True,
                persistent_workers=True if args.cores and args.cores > 0 else False,
                prefetch_factor=2 if args.cores and args.cores > 0 else None,
                drop_last=False,
            ),
            "train_dist": torch.utils.data.DataLoader(
                train_dist_datasets[i],
                batch_size=1,
                shuffle=False,
                num_workers=args.cores,
                pin_memory=True,
                persistent_workers=True if args.cores and args.cores > 0 else False,
                prefetch_factor=2 if args.cores and args.cores > 0 else None,
                drop_last=False,
            ),
        }
        all_loaders.append(loaders)

    # ---------------------------------------------------
    # 5. Make Trainer and start training
    # ---------------------------------------------------
    trainer = Trainer(
        models=models,
        optims=optimizers,
        scheds=schedulers,
        loaders=all_loaders,
        args=args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
