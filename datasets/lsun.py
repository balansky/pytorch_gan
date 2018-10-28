from datasets.dataset import Dataset
from torchvision import datasets, transforms


class Lsun(Dataset):

    def __init__(self, root='/tmp', train=True, image_transform=None, **kwargs):
        if not image_transform:
            image_transform = transforms.Compose([
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
            ])
        if train:
            cls = "train"
        else:
            cls = "val"
        lsun_dataset = datasets.LSUN(root=root, classes=cls,
                                    transform=image_transform)
        super(Lsun, self).__init__(lsun_dataset, **kwargs)