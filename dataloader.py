from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision.transforms import v2

class ImageDataset(Dataset):
    print('test')
    def __init__(self, image_dir, transform = None):
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        if self.transform is not None:
            image = self.transform(image)

        return image