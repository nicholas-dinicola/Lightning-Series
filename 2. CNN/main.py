import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from argparse import ArgumentParser
from typing import Optional, Any


class Dataset(pl.LightningDataModule):

    def __init__(self, data_path: str = './'):
        super().__init__()
        self.data_path = data_path
        self.transform = transforms.Compose(
            [transforms.ToTensor()]
        )

    def prepare_data(self) -> None:
        MNIST(root=self.data_path, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        mnist_all = MNIST(
            root=self.data_path,
            train=True,
            transform=self.transform,
            download=False
        )

        self.train, self.val = random_split(
            mnist_all, [55_000, 5_000], generator=torch.Generator().manual_seed(1)
        )

        self.test = MNIST(
            root=self.data_path,
            train=False,
            transform=self.transform,
            download=False
        )

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=64, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.train, batch_size=64, num_workers=4)


class CNN(pl.LightningModule):

    def __init__(
            self,
            cnn_out_channels=None,
            n_lables: int = 10
    ):
        super().__init__()

        if cnn_out_channels is None:
            cnn_out_channels = [16, 32, 64]
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        self.test_acc = Accuracy()

        in_channels = 1

        cnn_block = list()
        for out_channel in cnn_out_channels:
            cnn_block.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cnn_block.append(nn.ReLU())
            cnn_block.append(nn.MaxPool2d((2, 2)))
            in_channels = out_channel

        self.cnn_block = nn.Sequential(*cnn_block)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, n_lables)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.cnn_block(x)
        return self.head(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc.update(preds, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc", self.train_acc.compute())
        self.train_acc.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.valid_acc.update(preds, y)
        self.log("valid_loss", loss, prog_bar=True)
        return loss

    def validation_epoch_end(self, outs):
        self.log("valid_acc", self.valid_acc.compute(), prog_bar=True)
        self.valid_acc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc.compute(), prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer


def main(args: Any):
    dataset = Dataset()
    net = CNN()
    callbacks = [ModelCheckpoint(save_top_k=1, mode='max', monitor="valid_acc")]  # save top 1 model
    trainer = pl.Trainer(max_epochs=args.epochs, callbacks=callbacks, accelerator=args.accelerator, devices=1)
    trainer.fit(model=net, datamodule=dataset)
    trainer.test(model=net, datamodule=dataset, ckpt_path='best')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--accelerator", default="cpu", type=str)
    args = parser.parse_args()

    main(args)

    # $ python main.py --accelerator "cpu" --epochs 2




















