from tqdm import tqdm

import torch
import torch.nn as nn
import snntorch as snn
import snntorch.functional as SF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from snntorch import utils, surrogate

data_path = "~/robots/datasets/"
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)


class CorticalLayer(nn.Module):
    def __init__(self, input_size, output_size, beta=.99):
        super(CorticalLayer, self).__init__()

        self.layer = nn.Linear(input_size, output_size)
        self.layer_lif = snn.Leaky(spike_grad=snn.surrogate.fast_sigmoid(), beta=beta)
        self.mem = torch.zeros(output_size)

        self.spk = torch.zeros(output_size)
        self.error = None

        self.loss_fn = SF.mse_membrane_loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))

    def forward(self, input, error):
        # Feedforward through the layers
        # print(f'input: {input.shape}, error: {error.shape}, layer: {self.layer.weight.shape}')
        layer_out = self.layer(input)
        # print(f'layer_out.shape: {layer_out.shape}')
        spk, mem = self.layer_lif(layer_out, self.mem)
        self.spk = spk.detach()
        self.mem = mem.detach()

        print(f'mem.shape: {mem.shape}, error.shape: {error.shape}')
        loss_val = self.loss_fn(mem, error)
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()
        return self.spk


class InputArea(nn.Module):
    def __init__(self, input_size, layer_size, beta=.99):
        super(InputArea, self).__init__()

        self.granular = CorticalLayer(input_size + layer_size, layer_size)
        self.superficial = CorticalLayer(layer_size * 2, layer_size)
        self.deep = CorticalLayer(layer_size * 2, layer_size)
        self.next_area = None

    def forward(self, input):
        # print(f'self.deep.layer_lif.mem.shape: {self.deep.mem.shape}')
        granular_out = self.granular(torch.cat((input, self.deep.spk), 0), self.deep.mem)
        # print(f'granular_out.shape: {granular_out.shape}')
        # print(type(self.granular.mem))
        superficial_out = self.superficial(torch.cat((granular_out, self.next_area.superficial.spk), 0),
                                           self.next_area.superficial.mem)
        deep_out = self.deep(torch.cat((superficial_out, self.next_area.deep.spk), 0), self.next_area.deep.mem)
        return superficial_out

    def attach_above(self, area_above):
        self.next_area = area_above


class OutputArea(nn.Module):
    def __init__(self, layer_size, output_size, beta=.99):
        super(OutputArea, self).__init__()

        self.granular = CorticalLayer(layer_size * 2, layer_size)
        self.superficial = CorticalLayer(layer_size, output_size)
        self.deep = CorticalLayer(layer_size, layer_size)

        self.prev_area = None

    def forward(self, input, target):
        granular_out = self.granular(torch.cat((input, self.deep.spk), 0), self.deep.mem)
        superficial_out = self.superficial(granular_out, target)
        deep_out = self.deep(superficial_out, self.superficial.mem)
        return superficial_out

    def attach_below(self, area_below):
        self.prev_area = area_below


class MiddleArea(nn.Module):
    def __init__(self, layer_size, beta=.99):
        super(MiddleArea, self).__init__()

        self.granular = CorticalLayer(layer_size * 2, layer_size)
        self.superficial = CorticalLayer(layer_size * 2, layer_size)
        self.deep = CorticalLayer(layer_size * 2, layer_size)

        self.prev_area = None
        self.next_area = None

    def forward(self, input):
        granular_out = self.granular(torch.cat((input, self.deep.spk), 0), self.deep.mem)
        superficial_out = self.superficial(torch.cat((granular_out, self.next_area.superficial.spk), 0),
                                           self.next_area.superficial.mem)
        deep_out = self.deep(torch.cat((superficial_out, self.next_area.deep.spk), 0), self.next_area.deep.mem)
        return superficial_out

    def attach_above(self, area_above):
        self.next_area = area_above

    def attach_below(self, area_below):
        self.prev_area = area_below


class HierarchicalModel(nn.Module):
    def __init__(self, input_size, output_size, layer_size):
        super(HierarchicalModel, self).__init__()

        # Three cortical areas for demonstration
        self.input_area = InputArea(input_size, layer_size)
        self.middle_area = MiddleArea(layer_size)
        self.output_area = OutputArea(layer_size, output_size)

        self.input_area.attach_above(self.middle_area)
        self.middle_area.attach_above(self.output_area)
        self.middle_area.attach_below(self.input_area)
        self.output_area.attach_below(self.middle_area)

    def forward(self, input, target):
        input_layer_output = self.input_area(input)
        middle_layer_output = self.middle_area(input_layer_output)
        output_layer_output = self.output_area(middle_layer_output, target)


# Defining the function to generate the target tensor based on the digit label, total number of neurons, and number of classes.
def generate_target(digit, total_neurons, num_classes):
    # Calculate the number of neurons per class based on the total number of neurons and number of classes
    num_neurons_per_class = total_neurons // num_classes

    # Initialize the target tensor with zeros
    target = torch.zeros(total_neurons)

    # Find the start and end indices for the neurons corresponding to the given digit
    start_idx = digit * num_neurons_per_class
    end_idx = start_idx + num_neurons_per_class

    # Set the corresponding neurons to 1 (or any other activation level you prefer)
    target[start_idx:end_idx] = 1.0

    return target


def main():
    mnist_training_data = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )
    mnist_test_data = datasets.MNIST(
        data_path, train=False, download=True, transform=transform)

    mnist_training_data = utils.data_subset(mnist_training_data, 10)
    mnist_test_data = utils.data_subset(mnist_test_data, 10)

    # Initialize dataloaders
    train_loader = DataLoader(
        mnist_training_data,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        # prefetch_factor=8,
    )
    print("Train loader batches:", len(train_loader))
    test_loader = DataLoader(
        mnist_test_data, batch_size=1, shuffle=True, drop_last=True, num_workers=8
    )
    print("Test loader batches:", len(test_loader))

    cortical_stack = HierarchicalModel(28 * 28, 100, 100)

    for epoch in range(10):
        progress_bar = tqdm(iter(train_loader), total=len(train_loader))

        for data, targets in progress_bar:
            data = data.flatten()
            target_tensor = generate_target(targets.item(), 100, 10) # TODO: make this return spikes with a time dimension in spot 0
            #print(f'target_tensor: {target_tensor}')
            # data = data.view(-1, 28 * 28)

            # print(data.shape)
            for i in range(40):
                print(f'data: {data.shape}, target_tensor: {target_tensor.shape}')
                cortical_stack(data, target_tensor)
            # cortical_stack(data, target_tensor)


main()
