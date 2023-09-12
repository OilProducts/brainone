import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

import jax.numpy as jnp
from jax import random


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


def rate(data, num_steps=None, gain=1, offset=0, first_spike_time=0, time_var_input=False, rng=None):

    if first_spike_time < 0 or (num_steps is not None and num_steps < 0):
        raise Exception("``first_spike_time`` and ``num_steps`` cannot be negative.")

    if first_spike_time > (num_steps - 1):
        if num_steps:
            raise Exception(f"first_spike_time ({first_spike_time}) must be equal to or less than num_steps-1 ({num_steps-1}).")
        if not time_var_input:
            raise Exception("If the input data is time-varying, set ``time_var_input=True``. If the input data is not time-varying, ensure ``num_steps > 0``.")

    if first_spike_time > 0 and not time_var_input and not num_steps:
        raise Exception("``num_steps`` must be specified if both the input is not time-varying and ``first_spike_time`` is greater than 0.")

    if time_var_input and num_steps:
        raise Exception("``num_steps`` should not be specified if input is time-varying, i.e., ``time_var_input=True``. The first dimension of the input data + ``first_spike_time`` will determine ``num_steps``.")

    if rng is None:
        rng = random.PRNGKey(0)

    def rate_conv(input_data):
        """Convert input to Poisson spike train."""
        # Poisson spike train generation
        return random.bernoulli(rng, p=input_data).astype(jnp.float32)

    if time_var_input:
        spike_data = rate_conv(data)

        if first_spike_time > 0:
            spike_data = jnp.concatenate(
                [jnp.zeros((first_spike_time,) + data.shape[1:], dtype=jnp.float32), spike_data], axis=0
            )

    else:
        time_data_shape = (num_steps,) + data.shape
        time_data = jnp.tile(data, time_data_shape) * gain + offset

        spike_data = rate_conv(time_data)

        if first_spike_time > 0:
            spike_data = jnp.concatenate(
                [jnp.zeros((first_spike_time,) + data.shape, dtype=jnp.float32), spike_data], axis=0
            )

    return spike_data