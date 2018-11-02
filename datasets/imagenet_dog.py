
from datasets.dataset import Dataset
from torchvision import datasets, transforms
import torch
import random

class Crop(object):

    def __init__(self, augmentation=True, crop_ratio=0.9):
        self.augmentation = augmentation
        self.crop_ratio = crop_ratio

    def __call__(self, img):
        w, h = img.size
        short_side = h if h < w else w
        if self.augmentation:
            crop_size = int(short_side * self.crop_ratio)
            top = random.randint(0, h - crop_size - 1)
            left = random.randint(0, w - crop_size - 1)
        else:
            crop_size = short_side
            top = (h - crop_size) // 2
            left = (w - crop_size) // 2
        bottom = top + crop_size
        right = left + crop_size
        img = img.crop((left, top, right, bottom))
        return img

def add_noise(tensor):
    noise = torch.rand_like(tensor) * (1 / 128.)
    tensor += noise
    return tensor



class ImageNetDogDataset(Dataset):

    def __init__(self, root='/tmp', size=128, augmentation=True, image_transform=None, **kwargs):

        if not image_transform:
            image_transform = transforms.Compose([
                Crop(augmentation=augmentation),
                transforms.RandomHorizontalFlip(),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
                # transforms.Lambda(add_noise)
            ])

        dog_dataset = datasets.ImageFolder(root, image_transform)
        super(ImageNetDogDataset, self).__init__(dog_dataset, **kwargs)


