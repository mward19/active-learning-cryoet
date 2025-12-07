from transformers import AutoConfig, AutoModelForImageClassification, Trainer
from transformers import TrainingArguments
import torch
from torch.utils.data import Subset, DataLoader
from torch import nn

from abc import ABC, abstractmethod

class SubtomogramClassifier(nn.Module, ABC):
    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def train_step(self, batch):
        pass

    @abstractmethod
    def train_naive(self, dataset):
        pass

    # @abstractmethod
    # def train_active(self, dataset, strategy):
    #     pass

class ResNet3D(SubtomogramClassifier):
    def __init__(self, device=None):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        config = AutoConfig.from_pretrained(
            'nwirandx/medicalnet-resnet3d10',
            trust_remote_code=True,
            local_files_only=True
        )
        # use pretrained model
        self.model = AutoModelForImageClassification.from_pretrained(
            'nwirandx/medicalnet-resnet3d10',
            trust_remote_code=True,
            local_files_only=True
        ).to(self.device)

        self.loss_func = torch.nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.train_dataset = None
        self.eval_dataset = None

    def forward(self, x):
        return self.model(x).logits

    def init_training(self, train_dataset, eval_dataset):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

    def train_step(self, train_indices):
        """where train_indices is a 1D tensor of indices from train set to put in batch"""
        self.model.train()
        if self.train_dataset is None:
            raise RuntimeError("Training is not initialized")
        
        # Select samples
        subset = [self.train_dataset[i] for i in train_indices]

        # # Training step
        inputs = torch.stack([sample[0] for sample in subset]).to(self.device)
        labels = torch.stack([sample[1] for sample in subset]).to(self.device)
        # batch = {"pixel_values": inputs, "labels": labels}
        # from pdb import set_trace
        # set_trace()
        # loss = self.trainer.training_step(self.trainer.model, batch)
        
        # Forward pass
        logits = self.model(inputs).logits
        loss = self.loss_func(logits, labels)

        # Optimizer step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def validation_step(self, batch_size=8):
        if self.eval_dataset is None:
            raise RuntimeError("Training is not initialized")

        self.model.eval()
        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                preds = outputs.logits.argmax(dim=1)  # get predicted class
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        return accuracy

    def train_naive(self):
        # shuffled_indices = 
        pass

if __name__ == '__main__':
    from dataloader import SubtomogramPointDataset
    from torch.utils.data import random_split

    model = ResNet3D()

    print('Loaded model')

    dataset = SubtomogramPointDataset(
        r'/home/mward19/nobackup/autodelete/fm-data-2', 
        return_class=True, 
        max_tiles=100
    )

    model.init_training(dataset, None)
    model.train_step([0, 1, 2])



