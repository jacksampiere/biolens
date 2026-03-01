"""Train a constraint-aware 1D CNN encoder on synthetic EEG with mixed supervision.

This script builds a Lightning module with three parts: an EEG encoder, a projection
head for self-supervised learning, and a scalar monotonicity head trained on labeled
examples. Training uses a Gaussian noise augmentation for contrastive pairs and
combines contrastive + monotonicity losses, with the monotonicity weight linearly
warmed up by global step. Data is generated on the fly as synthetic, EEG-like signals
with partial labels.

Implementation notes:
- Ordered outcomes are integers ∈[1,30]
- EEG signals are single-channel and assumed to be of equal length
- Encoder is a 1D CNN; Adam optimizer
"""

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split


class ExperimentConfig:
    def __init__(
        self,
        cnn_config,
        seed=42,
        dataset_size=1000,
        label_frac=0.1,
        batch_size=128,
        num_workers=0,
        emb_dim=12,
        proj_dim=6,
        tau=0.1,  # temperature for InfoNCE loss
        beta_mono=1.0,  # sharpness of monotonicity loss
        lambda_mono=0.5,  # weight of monotonicity loss
        warmup_start_frac=0.0,  # fraction of total steps after which mono loss > 0
        warmup_end_frac=0.2,  # fraction of total steps for full warmup
        gen_noise_std=20,  # for EEG on the order of [-500, 500] µV
        aug_noise_std=20,
        lr=1e-3,
        max_epochs=10,
        C=1,
        T=2048,
        label_lower=1,
        label_upper=30,
    ):
        self.seed = seed
        self.dataset_size = dataset_size
        self.label_frac = label_frac
        self.C = C
        self.T = T
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.tau = tau
        self.beta_mono = beta_mono
        self.lambda_mono = lambda_mono
        self.warmup_start_frac = warmup_start_frac
        self.warmup_end_frac = warmup_end_frac
        self.gen_noise_std = gen_noise_std
        self.aug_noise_std = aug_noise_std
        self.lr = lr
        self.max_epochs = max_epochs
        self.cnn_config = cnn_config
        self.label_lower = label_lower
        self.label_upper = label_upper


class CNNConfig:
    def __init__(
        self,
        conv_channels,
        kernel_sizes,
    ):
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes


class EEGEncoder1DCNN(nn.Module):
    def __init__(
        self, in_channels, conv_channels, kernel_sizes, emb_dim, adaptive_pool_size=1
    ):
        super().__init__()

        if len(conv_channels) != len(kernel_sizes):
            raise ValueError(
                "`conv_channels` and `kernel_sizes` must have the same length."
            )

        # Accumulate convolutional and pooling layers
        blocks = []
        current_dim = in_channels
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv_layer = nn.Conv1d(
                in_channels=current_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
            )
            blocks.append(conv_layer)
            blocks.append(nn.ReLU())
            blocks.append(nn.MaxPool1d(kernel_size=kernel_size))
            current_dim = out_channels

        self.layers = nn.Sequential(*blocks)

        # Pool + project
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=adaptive_pool_size)
        self.embedding_layer = nn.Linear(
            in_features=current_dim * adaptive_pool_size, out_features=emb_dim
        )  # number of input features depends on final conv/pool layers and global pooling

    def forward(self, x):
        x = self.layers(x)
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.embedding_layer(x)
        return x


class SSLProjectionHead(nn.Module):
    def __init__(self, emb_dim, proj_dim):
        super().__init__()
        self.projection = nn.Linear(in_features=emb_dim, out_features=proj_dim)

    def forward(self, embeddings):
        return self.projection(embeddings)


class MonotonicityHead(nn.Module):
    def __init__(self, emb_dim, proj_dim=1):
        super().__init__()
        self.projection = nn.Linear(in_features=emb_dim, out_features=proj_dim)

    def forward(self, embeddings):
        return self.projection(embeddings)


class ConstraintAwareEEGEncoder(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.warmup_end_frac <= config.warmup_start_frac:
            raise ValueError("Warmup end must be > warmup start")
        # Placeholders; calculate true steps in on_fit_start()
        self.warmup_start_step = 0
        self.warmup_end_step = 1

        eeg_encoder = EEGEncoder1DCNN(
            in_channels=config.C,
            conv_channels=config.cnn_config.conv_channels,
            kernel_sizes=config.cnn_config.kernel_sizes,
            emb_dim=config.emb_dim,
        )
        self.eeg_encoder = eeg_encoder

        projection_head = SSLProjectionHead(
            emb_dim=config.emb_dim,
            proj_dim=config.proj_dim,
        )
        self.projection_head = projection_head

        monotonicity_head = MonotonicityHead(
            emb_dim=config.emb_dim,
            proj_dim=1,  # scalar readout
        )
        self.monotonicity_head = monotonicity_head

    def on_fit_start(self):
        """Computes warmup steps from configured fractions."""
        total_steps = self.trainer.estimated_stepping_batches
        self.warmup_start_step = int(self.config.warmup_start_frac * total_steps)
        self.warmup_end_step = int(self.config.warmup_end_frac * total_steps)

    def forward(self, x):
        z = self.eeg_encoder(x)
        h = self.projection_head(z)
        s = self.monotonicity_head(z).squeeze(
            -1
        )  # want 1D for pairwise comparisons in monotonicity loss
        return h, s

    def training_step(self, batch, batch_idx):
        # SSL loss (InfoNCE)
        x1 = batch["x"]
        x2 = gaussian_noise_augment(batch["x"], self.config.aug_noise_std)
        p1, s1 = self(x1)
        p2, _ = self(x2)
        loss_ssl = infonce_loss(p1, p2, self.config.tau)

        # Monotonicity loss
        label_mask = batch["is_labeled"]
        y_labeled = batch["y"][label_mask]
        s_labeled = s1[label_mask]
        loss_mono = monotonicity_loss(y_labeled, s_labeled, self.config.beta_mono)

        # Total loss (with warmup)
        loss = loss_ssl + self.config.lambda_mono * self.eta() * loss_mono
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # SSL loss (InfoNCE)
        x1 = batch["x"]
        x2 = gaussian_noise_augment(batch["x"], self.config.aug_noise_std)
        p1, s1 = self(x1)
        p2, _ = self(x2)
        loss_ssl = infonce_loss(p1, p2, self.config.tau)

        # Monotonicity loss
        label_mask = batch["is_labeled"]
        y_labeled = batch["y"][label_mask]
        s_labeled = s1[label_mask]
        loss_mono = monotonicity_loss(y_labeled, s_labeled, self.config.beta_mono)

        # Total loss (with warmup)
        loss = loss_ssl + self.config.lambda_mono * self.eta() * loss_mono
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.config.lr)

    def eta(self):
        """Warmstart schedule for monotonicity loss weight."""
        step = self.global_step
        start = self.warmup_start_step
        end = self.warmup_end_step

        if step <= start:
            return 0
        if step >= end:
            return 1

        return float(step - start) / float(end - start)


def gaussian_noise_augment(x, noise_std):
    return x + torch.randn_like(x) * noise_std


def infonce_loss(p1, p2, tau):
    # L2 norm + similarities
    p1 = F.normalize(p1, p=2, dim=1)
    p2 = F.normalize(p2, p=2, dim=1)
    logits = (p1 @ p2.T) / tau  # temperature-scaled pairwise similarities
    # Encourage each view to be predictive of the other
    targets = torch.arange(
        p1.size(0), device=p1.device
    )  # correct class index (vector i from p1 is a positive pair with vector i from p2)
    loss_12 = F.cross_entropy(logits, targets)
    loss_21 = F.cross_entropy(logits.T, targets)
    return 0.5 * (loss_12 + loss_21)


def monotonicity_loss(y, s, beta):
    n_labeled = y.numel()  # only pass labeled pairs to loss calculation
    if n_labeled < 2:
        return torch.tensor(0.0, device=y.device)
    # All ij pairs --> all upper triangular elements, but not the diagonal (hence offset=1)
    ii, jj = torch.triu_indices(n_labeled, n_labeled, offset=1, device=y.device)
    dy = y[ii] - y[jj]
    ds = s[ii] - s[jj]
    loss_mono = F.softplus(-beta * (dy * ds)).mean()
    return loss_mono


class SyntheticEEGDataset(Dataset):
    def __init__(
        self,
        dataset_size,
        C,
        T,
        gen_noise_std,
        label_frac,
        label_lower,
        label_upper,
        label_mask=-1,
    ):
        super().__init__()
        self.dataset_size = dataset_size
        self.C = C
        self.T = T
        self.gen_noise_std = gen_noise_std
        self.label_mask = label_mask

        signals = self._gen_signals()
        self.signals = signals

        y = torch.full((self.dataset_size,), -1, dtype=torch.long)
        labeled = torch.rand(self.dataset_size) < label_frac
        n_labeled = labeled.sum()
        y[labeled] = torch.randint(label_lower, label_upper + 1, size=(n_labeled,))
        self.y = y

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return {
            "x": self.signals[idx, ...],
            "y": self.y[idx],
            "is_labeled": self.y[idx] != self.label_mask,
        }

    def _gen_signals(self):
        # Time axis for broadcasting
        t = torch.linspace(0.0, 1.0, self.T, dtype=torch.float32).view(1, 1, self.T)
        # Each channel gets a random frequency between 1 and 60 Hz
        freq = torch.rand(self.dataset_size, self.C, 1) * (60 - 1) + 1
        # Each channel gets a random phase
        phase = torch.rand(self.dataset_size, self.C, 1) * (2 * torch.pi)
        # Each channel gets a random amplitude scale between 200 and 500 µV
        amplitude = torch.rand(self.dataset_size, self.C, 1) * 300 + 200
        # Build signals given frequency + phase + amplitude
        signal = amplitude * torch.sin((2 * torch.pi * freq * t) + phase)
        # Add some noise
        noise = (
            torch.randn(self.dataset_size, self.C, self.T, dtype=torch.float32)
            * self.gen_noise_std
        )
        x = signal + noise
        # Clip and scale for numerical stability
        q = torch.quantile(x.abs(), 0.95, dim=-1, keepdim=True).clamp_min(1e-6)
        x = torch.clamp(x, min=-q, max=q)
        x = x / q
        return x


if __name__ == "__main__":
    torch.manual_seed(42)

    # Setup
    cnn_config = CNNConfig(conv_channels=[2, 4, 6], kernel_sizes=[7, 5, 3])
    config = ExperimentConfig(cnn_config=cnn_config)
    constraint_aware_model = ConstraintAwareEEGEncoder(config)
    dataset = SyntheticEEGDataset(
        config.dataset_size,
        config.C,
        config.T,
        config.gen_noise_std,
        config.label_frac,
        config.label_lower,
        config.label_upper,
    )

    # TVT split
    p_train, p_val = 0.8, 0.2
    n_train = int(p_train * config.dataset_size)
    n_val = config.dataset_size - n_train
    train_dataset, val_dataset = random_split(dataset, lengths=[n_train, n_val])

    # Dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=config.batch_size, num_workers=config.num_workers
    )

    # Train
    ckpt = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        filename="best_model_{epoch}_{val_loss:.4f}",
        auto_insert_metric_name=True,
    )
    trainer = L.Trainer(max_epochs=config.max_epochs, callbacks=[ckpt])
    trainer.fit(
        constraint_aware_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Load best model
    best_constraint_aware_model = ConstraintAwareEEGEncoder.load_from_checkpoint(
        ckpt.best_model_path,
        config=config,
    )
