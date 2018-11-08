from datasets.dataset import Dataset
from torchvision import datasets, transforms


class Cifar10(Dataset):

    def __init__(self, root='/tmp', train=True, image_transform=None, **kwargs):
        if not image_transform:
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
                # transforms.Normalize(mean=[.485, .456, .406], std=[0.229, 0.224, 0.225])
            ])

        cifar_dataset = datasets.CIFAR10(root=root, train=train, download=True,
                                         transform=image_transform)
        super(Cifar10, self).__init__(cifar_dataset, **kwargs)
