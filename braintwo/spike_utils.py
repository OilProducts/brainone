import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def get_mnist_dataloaders(shrink_factor=1, batch_size=64, shuffle=True, num_workers=0):
    """
    Returns a dataloader for MNIST dataset.

    Parameters:
    - shrink_factor (int): Factor by which dataset size should be reduced. E.g., a factor of 10 means dataset will be 1/10th of original size.
    - batch_size (int): Batch size for the dataloader.
    - shuffle (bool): Whether to shuffle the data.
    - num_workers (int): Number of subprocesses to use for data loading.

    Returns:
    - DataLoader object.
    """

    # Define the transformation - Convert data to tensor & normalize
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,))
    ])

    # Load training MNIST dataset
    mnist_train_dataset = datasets.MNIST(root='./data',
                                         train=True,
                                         transform=transform,
                                         download=True)

    # If shrink factor is provided and greater than 1, reduce the dataset size
    if shrink_factor > 1:
        total_samples = len(mnist_train_dataset)
        indices = list(range(0, total_samples, shrink_factor))
        mnist_train_dataset = Subset(mnist_train_dataset, indices)

    # Create dataloader
    mnist_train_dataloader = DataLoader(mnist_train_dataset,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_workers=num_workers,
                                        drop_last=True)

    # Load test MNIST dataset
    mnist_test_dataset = datasets.MNIST(root='./data',
                                        train=False,
                                        transform=transform,
                                        download=True)

    # If shrink factor is provided and greater than 1, reduce the dataset size
    if shrink_factor > 1:
        total_samples = len(mnist_test_dataset)
        indices = list(range(0, total_samples, shrink_factor))
        mnist_test_dataset = Subset(mnist_test_dataset, indices)

    # Create dataloader
    mnist_test_dataloader = DataLoader(mnist_test_dataset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       drop_last=True)

    return mnist_train_dataloader, mnist_test_dataloader