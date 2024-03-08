import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import os
from resnet_model import *
from torchvision.transforms import v2

"""from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)"""

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 4
IMAGE_HEIGHT = 256  
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

DATA_DIR = "Data/"

"""TRAIN_COVID_IMG_DIR = "Data/covid/img"
TRAIN_COVID_MASK_DIR = "Data/covid/mask"
TRAIN_NORM_IMG_DIR = "Data/normal/img"
TRAIN_NORM_MASK_DIR = "Data/normal/mask"""


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, targets, labels) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            #print(predictions)
            #print(targets)
            loss = loss_fn(predictions, labels)
            print(f"Loss: {loss}")

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    # Load the model
    classes = os.listdir(DATA_DIR)
    num_classes = len(classes)
    model = test()
    print(model)

    # Transforms
    train_transform = v2.Compose([
        v2.ToTensor(),
        v2.RandomCrop(32, 4),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.1),
        v2.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std= np.sqrt([1.0, 1.0, 1.0]) # variance is std**2
        )
        ])

if __name__ == "__main__":
    main()