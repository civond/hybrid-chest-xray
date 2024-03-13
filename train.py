import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import torchvision.models as models
from torchvision.transforms import v2

from resnet import *
from utils import *

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 10
NUM_WORKERS = 4
IMAGE_HEIGHT = 256  
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

DATA_DIR = "Data/"

# Directories
TRAIN_COVID_IMG_DIR = "Data/covid/img"
TRAIN_NORM_IMG_DIR = "Data/normal/img"
TEST_COVID_IMG_DIR = "test_data/covid/img"
TEST_NORM_IMG_DIR = "test_data/normal/img"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, labels) in enumerate(loop):
        data = data.to(device=DEVICE)
        labels = labels.to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, labels)

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

    #model_eff = models.efficientnet_b0(pretrained=True).to(DEVICE)
    #model_eff.classifier[1] = nn.Linear(1280, num_classes)
    #model = ResNet50(img_channel=3, num_classes=2).to(DEVICE)

    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(1280, num_classes)
    model = model.to(DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()

    # Transforms
    train_transform = v2.Compose([
        v2.ToTensor(),
        v2.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.1),
        v2.RandomRotation(degrees=35),
        v2.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std= np.sqrt([1.0, 1.0, 1.0]) # variance is std**2
        )
        ])
    
    test_transform = v2.Compose([
            v2.ToTensor(),
            v2.Resize((IMAGE_HEIGHT,IMAGE_WIDTH)),
            v2.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std= np.sqrt([1.0, 1.0, 1.0]) # variance is std**2
                )
        ])
    train = True
    train_loader = get_loader(
            TRAIN_COVID_IMG_DIR,
            TRAIN_NORM_IMG_DIR,
            BATCH_SIZE,
            train_transform,
            NUM_WORKERS,
            train,
            PIN_MEMORY
        )
    
    train = False
    test_loader = get_loader(
            TEST_COVID_IMG_DIR,
            TEST_NORM_IMG_DIR,
            BATCH_SIZE,
            test_transform,
            NUM_WORKERS,
            train,
            PIN_MEMORY
        )
    
    if LOAD_MODEL: 
        print("Loading model.")
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch: {epoch}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        save_checkpoint(checkpoint)
        accuracy = check_accuracy(test_loader, 
                                  model, 
                                  device=DEVICE)



if __name__ == "__main__":
    main()