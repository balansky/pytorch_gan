from datasets.dataset import Dataset
from torchvision import datasets, transforms


class Stl10(Dataset):

    def __init__(self, root='/tmp', train=True, image_transform=None, **kwargs):
        if not image_transform:
            image_transform = transforms.Compose([
                transforms.RandomCrop(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])
        if train:
            split = 'train'
        else:
            split = 'test'
        cifar_dataset = datasets.STL10(root=root, split=split, download=True,
                                       transform=image_transform)
        super(Stl10, self).__init__(cifar_dataset, **kwargs)