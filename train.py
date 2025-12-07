import torch
from transformers import Trainer

from dataloader import SubtomogramPointDataset
from load_models import load_resnet

def train(model, dataset):
    pass

if __name__ == '__main__':
    dataset = SubtomogramPointDataset(
        r'/home/mward19/nobackup/autodelete/fm-data-2', 
        bool_mode=False, 
        max_tiles=1000
    )

    model = load_resnet()

    train(model, dataset)

