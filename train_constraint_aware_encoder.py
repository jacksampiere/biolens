# TODO: Summary of script + list assumptions
# - All signals of same length
# - Need to implement clip and scale / standardization? How?

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split


# TODO: Add monotonicity components + instantiate by reading from config file
class ExperimentConfig:
    def __init__(
        self,
        cnn_config,
        seed=42,
        dataset_size=1000,
        label_frac=0.1,
        batch_size=32,
        num_workers=0,
        emb_dim=12,
        proj_dim=6,
        tau=0.1,  # temperature for InfoNCE loss
        gen_noise_std=20,  # for EEG on the order of [-500, 500] µV
        aug_noise_std=20,
        lr=1e-3,
        max_epochs=10,
        C=1,
        T=2048,
        labels=list(range(1, 31)),
    ):
        self.seed = seed
        self.dataset_size = dataset_size
        self.label_frac = label_frac
        self.C = C
        self.T = T
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.tau = tau
        self.gen_noise_std = gen_noise_std
        self.aug_noise_std = aug_noise_std
        self.lr = lr
        self.max_epochs = max_epochs
        self.cnn_config = cnn_config


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

        # Accumlate convolutional and pooling layers
        self.layers = nn.ModuleList()
        current_dim = in_channels
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            conv_layer = nn.Conv1d(
                in_channels=current_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
            )
            pooling = nn.MaxPool1d(kernel_size=kernel_size)
            self.layers.append(conv_layer)
            self.layers.append(pooling)
            current_dim = out_channels

        # Pool + project
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=adaptive_pool_size)
        self.embedding_layer = nn.Linear(
            in_features=current_dim * adaptive_pool_size, out_features=emb_dim
        )  # number of input features depends on final conv/pool layers and global pooling

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        x = self.global_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.embedding_layer(x)
        return x


class SSLProjectionHead(nn.Module):
    def __init__(self, emb_dim, proj_dim):
        super().__init__()
        self.proj_dim = proj_dim
        self.projection = nn.Linear(in_features=emb_dim, out_features=proj_dim)

    def forward(self, embeddings):
        return self.projection(embeddings)


class ConstraintAwareEEGEncoder(L.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config

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

    def forward(self, x):
        z = self.eeg_encoder(x)
        h = self.projection_head(z)
        return h

    def training_step(self, batch, batch_idx):
        x1 = batch
        x2 = gaussian_noise_augment(batch, self.config.aug_noise_std)
        p1, p2 = self(x1), self(x2)
        loss = infonce_loss(p1, p2, self.config.tau)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x1 = batch
        x2 = gaussian_noise_augment(batch, self.config.aug_noise_std)
        p1, p2 = self(x1), self(x2)
        loss = infonce_loss(p1, p2, self.config.tau)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.config.lr)

    # def monotonicity_loss(self):
    #     pass

    # def eta(self):
    #     """Warmstart schedule for monotonicity loss weight."""
    #     pass


def gaussian_noise_augment(x, noise_std):
    return x + torch.randn_like(x) * noise_std


"""
Intuition:
- Build a B x B score table: “who matches who.”
- Diagonal = true matches, off-diagonal = impostors.
- For each row, softmax turns scores into probabilities over candidates.
- Loss says: “put high probability on the diagonal.”
- Do it both directions and average.
"""


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


# TODO: Update to include monotonicity components
class SyntheticEEGDataset(Dataset):
    def __init__(self, dataset_size, C, T, gen_noise_std):
        super().__init__()
        self.dataset_size = dataset_size
        self.C = C
        self.T = T
        self.gen_noise_std = gen_noise_std
        signals = self._gen_signals()
        self.signals = signals

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.signals[idx, ...]
        # return {
        #     "x": self.signals[idx, ...],
        #     "y": label,
        #     "is_labeled": is_labeled
        # }

    def _gen_signals(self):
        # Time axis for broadcasting
        t = torch.linspace(0.0, 1.0, self.T, dtype=torch.float32).view(1, 1, self.T)
        # Each channel gets a random frequency between 1 and 60 Hz
        freq = torch.rand(self.dataset_size, self.C, 1) * (60 - 1) + 1
        # Each channel gets a random phase
        phase = torch.rand(self.dataset_size, self.C, 1) * (2 * torch.pi)
        # Each channel gets a random amplitude scale between 200 and 500 µV
        amplitude = torch.rand(self.dataset_size, self.C, 1) * 300 + 200
        # Build singals given frequency + phase + amplitude
        signal = amplitude * torch.sin((2 * torch.pi * freq * t) + phase)
        # Add some noise
        noise = (
            torch.randn(self.dataset_size, self.C, self.T, dtype=torch.float32)
            * self.gen_noise_std
        )
        return signal + noise

    # def _sample_label_mask(self):
    #     pass


if __name__ == "__main__":
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup
    cnn_config = CNNConfig(conv_channels=[2, 4, 6], kernel_sizes=[7, 5, 3])
    config = ExperimentConfig(cnn_config=cnn_config)
    constaint_aware_model = ConstraintAwareEEGEncoder(config)
    dataset = SyntheticEEGDataset(
        config.dataset_size,
        config.C,
        config.T,
        config.gen_noise_std,
    )

    # TVT split
    p_train, p_val = 0.8, 0.2
    n_train = int(p_train * config.dataset_size)
    n_val = config.dataset_size - n_train
    train_dataset, val_dataset = random_split(dataset, lengths=[n_train, n_val])

    # Dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=config.num_workers
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
        constaint_aware_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # Load best model
    best_constraint_aware_model = ConstraintAwareEEGEncoder.load_from_checkpoint(
        ckpt.best_model_path,
        config=config,
    )
