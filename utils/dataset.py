import torch
from torch.utils.data import Dataset

from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10
import albumentations as alb
from albumentations.pytorch import ToTensorV2

from PIL import Image 
import numpy as np
import pickle


class FLDataset(Dataset):
    """
    create a cumtomized dataset
    :param transform: transforms include extra data augmentation, basic transforms like normalize and totensor is already implemented
    """
    def __init__(self, path, dataset, train, num_classes, transform = None, target_transform=None):
        super(FLDataset, self).__init__()

        self.index_per_class = [[] for i in range(num_classes)]
        self.num_per_class = [0 for i in range(num_classes)]
        self.images = []
        self.labels = []
        self.mean = None
        self.std = None

        self.num_classes = num_classes

        self.train = train

        if dataset == "MNIST":
            # path: ./data/
            self._dataset = MNIST(path, train = self.train, download=False)
            self.images, self.labels = self._dataset.data, self._dataset.targets
            self.images = self.images.unsqueeze(1) / 255.0
            self.mean = (0.13066047863205824,)
            self.std = (0.3015042652604464,)
            
        elif dataset == "cifar10":
            # path: ./data/
            self._dataset = CIFAR10(path, train = self.train, download=False)
            self.images, self.labels = self._dataset.data, self._dataset.targets
            self.images = torch.from_numpy(self.images).transpose(1,3) / 255.0
            self.mean = (0.49139969396462474, 0.48215842334762077, 0.4465309297941824)
            self.std = (0.20230092356590415, 0.19941281395938265, 0.2009616141524328)
            # print(self.images.shape)

        else:
            ## TODO
            ...

        # get index of images in each category
        for i in range(len(self.images)):
            self.index_per_class[int(self.labels[i])].append(i)
            self.num_per_class[self.labels[i]] += 1
        #  MNIST is imbalanced and we set the number of each category to be
        #  the minimum number of data in 10 categories
        min_num = min(self.num_per_class)  #5421 for MNIST training set

        for i in range(self.num_classes):
            self.index_per_class[i] = self.index_per_class[i][:min_num]
        # prepare transform for images and labels
        self.target_transform = target_transform
        if transform == None:
            if self.mean == None:
                self.mean, self.std = computeMeanStd(self.images)

            self.transform = transforms.Compose([
                transforms.Normalize(mean = self.mean, std = self.std),
            ])
        else:
            # compose transform
            self.transform = transform.Compose([
                transform,
                transforms.Normalize(mean = self.mean, std = self.std),
            ])
 
    def __getitem__(self, index):

        image, label = self.images[index], self.labels[index]
        image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return image, label
    
    def __len__(self):
        return len(self.images)


def computeMeanStd(images: torch.Tensor):

    batch, channel = images.shape[0], images.shape[1]

    mean = [0 for i in range(channel)]
    std = [0 for i in range(channel)]

    for i in range(batch):
        for j in range(channel):
            mean[j] += images[i, j, :, :].mean().item() / batch
            std[j] += images[i, j, :, :].std().item() / batch

    return mean, std


if __name__ == "__main__":

    dataset = FLDataset(path="/mnt/traffic/leijiachen/data",
              dataset = "MNIST",
              train = False,
              num_classes = 10,
              )
    print(dataset.num_per_class)
    # print(dataset[1][0])
    # img = transforms.ToPILImage()(dataset[0][0])
    # img.save("output.png")