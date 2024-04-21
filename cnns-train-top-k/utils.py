import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms


def get_data_loaders(dataset_path, batch_size=32):
    """
    Gets the data loaders for the training, test, and stylised test datasets.

    Args:
        dataset_path (str): The path to the dataset. The dataset should have the following structure:
            dataset_path
            ├── train
            │   ├── class1
            │   │   ├── img1.jpg
            │   │   ├── img2.jpg
            │   │   └── ...
            │   ├── class2
            │   │   ├── img1.jpg
            │   │   ├── img2.jpg
            │   │   └── ...
            │   └── ...
            ├── test ...
            └── stylised_test ...

        batch_size (int): The batch size. (default: 32)
    """
    # The transform_mean and transform_std values are used to normalise the image data.
    # The is usually done to train a model on ImageNet dataset.
    transform_mean = np.array([0.485, 0.456, 0.406])
    transform_std = np.array([0.229, 0.224, 0.225])

    # The transform_train and transform_test transformations are used to preprocess the training and test datasets.
    # The transformations are the same as the ones used to train the ResNet model on the ImageNet dataset.
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_mean, std=transform_std),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=transform_mean, std=transform_std),
        ]
    )

    # The trainset
    trainset = torchvision.datasets.ImageFolder(
        root="{}/train".format(dataset_path), transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    # The testset
    testset = torchvision.datasets.ImageFolder(
        root="{}/test".format(dataset_path), transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False
    )
    # The stylisedtestset
    stylised_testset = torchvision.datasets.ImageFolder(
        root="{}/stylised_test".format(dataset_path), transform=transform_test
    )
    stylised_testloader = torch.utils.data.DataLoader(
        stylised_testset, batch_size=batch_size, shuffle=False
    )

    return trainloader, testloader, stylised_testloader
