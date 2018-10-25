
from datasets.dataset import Dataset
from torchvision import datasets, transforms

class ImageNetDogDataset(Dataset):

    def __init__(self, root='/tmp', image_transform=None, **kwargs):
        if not image_transform:
            image_transform = transforms.Compose([
                transforms.RandomResizedCrop(128),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])

        dog_dataset = datasets.ImageFolder(root, image_transform)
        super(ImageNetDogDataset, self).__init__(dog_dataset, **kwargs)


