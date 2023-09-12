import cProfile

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
    def __init__(self, input_size, output_size, beta=.99, loss_fn=nn.MSELoss()):
        super(CorticalLayer, self).__init__()

        self.layer = nn.Linear(input_size, output_size)
        self.layer_lif = snn.Leaky(spike_grad=snn.surrogate.fast_sigmoid(), beta=beta)
        self.mem = torch.zeros(output_size)

        self.spk = torch.zeros(1, output_size)
        self.error = None

        self.loss_fn = loss_fn
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.999))
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'initialized parameters: {num_params}')

    def forward(self, input, error):
        print(f'input: {input.shape}')
        # Feedforward through the layers
        input = input.detach()
        error = error.detach()
        layer_out = self.layer(input)
        spk, mem = self.layer_lif(layer_out, self.mem)
        self.spk = spk
        self.mem = mem.squeeze()
        #self.mem = mem.detach().squeeze

        loss_val = self.loss_fn(mem, error)
        self.optimizer.zero_grad()
        loss_val.backward()
        self.optimizer.step()
        return self.spk


def custom_loss(output_mem, target_tensor):
    num_classes = 10  # Number of classes
    num_neurons_per_class = output_mem.shape[1] // num_classes  # Number of neurons per class

    # Aggregate the membrane potentials for each class
    classwise_sum = torch.zeros(num_classes)
    for i in range(num_classes):
        start_idx = i * num_neurons_per_class
        end_idx = start_idx + num_neurons_per_class
        classwise_sum[i] = output_mem[0, start_idx:end_idx].sum()

    # Apply softmax activation
    softmax_output = torch.nn.functional.softmax(classwise_sum, dim=0)

    # Calculate the target class index
    target_class_idx = torch.argmax(target_tensor)

    # Compute cross-entropy loss
    loss = -torch.log(softmax_output[target_class_idx])

    return loss


class InputArea(nn.Module):
    def __init__(self, input_size, layer_size, beta=.99):
        super(InputArea, self).__init__()

        self.granular = CorticalLayer(input_size + layer_size, layer_size)
        self.superficial = CorticalLayer(layer_size * 2, layer_size)
        self.deep = CorticalLayer(layer_size * 2, layer_size)
        self.next_area = None

    def forward(self, input):
        granular_out = self.granular(torch.cat((input, self.deep.spk), 1), self.deep.mem)

        superficial_out = self.superficial(torch.cat((granular_out, self.next_area.superficial.spk), 1),
                                           self.next_area.superficial.mem)
        deep_out = self.deep(torch.cat((superficial_out, self.next_area.deep.spk), 1), self.next_area.deep.mem)
        return superficial_out

    def attach_above(self, area_above):
        self.next_area = area_above


class MiddleArea(nn.Module):
    def __init__(self, layer_size, beta=.99):
        super(MiddleArea, self).__init__()

        self.granular = CorticalLayer(layer_size * 2, layer_size)
        self.superficial = CorticalLayer(layer_size * 2, layer_size)
        self.deep = CorticalLayer(layer_size * 2, layer_size)

        self.prev_area = None
        self.next_area = None

    def forward(self, input):
        granular_out = self.granular(torch.cat((input, self.deep.spk), 1), self.deep.mem)
        superficial_out = self.superficial(torch.cat((granular_out, self.next_area.superficial.spk), 1),
                                           self.next_area.superficial.mem)
        deep_out = self.deep(torch.cat((superficial_out, self.next_area.deep.spk), 1), self.next_area.deep.mem)
        return superficial_out

    def attach_above(self, area_above):
        self.next_area = area_above

    def attach_below(self, area_below):
        self.prev_area = area_below


class OutputArea(nn.Module):
    def __init__(self, layer_size, output_size, beta=.99):
        super(OutputArea, self).__init__()

        self.granular = CorticalLayer(layer_size * 2, layer_size)
        self.superficial = CorticalLayer(layer_size, output_size, loss_fn=custom_loss)
        self.deep = CorticalLayer(layer_size, layer_size)

        self.prev_area = None

    def forward(self, input, target):
        granular_out = self.granular(torch.cat((input, self.deep.spk), 1), self.deep.mem)
        superficial_out = self.superficial(granular_out, target)
        deep_out = self.deep(superficial_out, self.superficial.mem)
        return superficial_out

    def attach_below(self, area_below):
        self.prev_area = area_below


class HierarchicalModel(nn.Module):
    def __init__(self, input_size, output_size, layer_size):
        super(HierarchicalModel, self).__init__()

        # Three cortical areas for demonstration
        self.input_area = InputArea(input_size, layer_size)
        self.middle_one = MiddleArea(layer_size)
        self.middle_two = MiddleArea(layer_size)
        self.middle_three = MiddleArea(layer_size)
        self.middle_four = MiddleArea(layer_size)
        self.output_area = OutputArea(layer_size, output_size)

        self.input_area.attach_above(self.middle_one)
        self.middle_one.attach_above(self.middle_two)
        self.middle_two.attach_above(self.middle_three)
        self.middle_three.attach_above(self.middle_four)
        self.middle_four.attach_above(self.output_area)
        self.output_area.attach_below(self.middle_four)
        self.middle_four.attach_below(self.middle_three)
        self.middle_three.attach_below(self.middle_two)
        self.middle_two.attach_below(self.middle_one)
        self.middle_one.attach_below(self.input_area)

    def forward(self, input, target):
        target_tensor = generate_target(target.item(), 100, 10)
        input_layer_output = self.input_area(input)
        middle_one_output = self.middle_one(input_layer_output)
        middle_two_output = self.middle_two(middle_one_output)
        middle_three_output = self.middle_three(middle_two_output)
        middle_four_output = self.middle_four(middle_three_output)
        output_layer_output = self.output_area(middle_four_output, target)
        return output_layer_output


# Defining the function to generate the target tensor based on the digit label, total number of neurons, and number of classes.
def generate_target(digit, total_neurons, num_classes):
    # Calculate the number of neurons per class based on the total number of neurons and number of classes
    num_neurons_per_class = total_neurons // num_classes

    # Initialize the target tensor with zeros
    target = torch.zeros(1, total_neurons)
    # target = torch.zeros(total_neurons)

    # Find the start and end indices for the neurons corresponding to the given digit
    start_idx = digit * num_neurons_per_class
    end_idx = start_idx + num_neurons_per_class

    # Set the corresponding neurons to 1 (or any other activation level you prefer)
    target[0, start_idx:end_idx] = 1.0

    return target


def predict_class_over_time(output_tensor, time_steps=8):
    # Reshape the tensor to [steps, 1, 10, 10]
    reshaped_tensor = output_tensor.view(time_steps, 1, 10, -1)

    # Sum along the last dimension to get total activity for each class at each time step
    class_activity = torch.sum(reshaped_tensor, dim=3)

    # Sum along the time dimension
    total_class_activity = torch.sum(class_activity, dim=0)

    # Find the index of the maximum value to get the predicted class
    _, predicted_class = torch.max(total_class_activity, dim=1)

    return predicted_class.item()


def predict_class(output_tensor):
    # Reshape the tensor to [1, 10, 10]
    reshaped_tensor = output_tensor.view(1, 10, -1)

    # Sum along the third dimension to get total activity for each class
    class_activity = torch.sum(reshaped_tensor, dim=2)

    # Find the index of the maximum value to get the predicted class
    _, predicted_class = torch.max(class_activity, dim=1)

    return predicted_class.item()


def main():
    mnist_training_data = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )
    mnist_test_data = datasets.MNIST(
        data_path, train=False, download=True, transform=transform)

    mnist_training_data = utils.data_subset(mnist_training_data, 1000)
    mnist_test_data = utils.data_subset(mnist_test_data, 1000)

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

    for epoch in range(1):
        progress_bar = tqdm(iter(train_loader), total=len(train_loader))

        samples_seen = 0
        num_correct = 0

        for data, targets in progress_bar:
            data = data.flatten()

            target_tensor = generate_target(targets.item(), 100,
                                            10)  # TODO: make this return spikes with a time dimension in spot 0
            # print(f'target_tensor: {target_tensor}')
            # data = data.view(-1, 28 * 28)

            # print(data.shape)
            output = []
            steps = 8
            for i in range(steps):
                spikes = snn.spikegen.rate(data, 1)
                activations = cortical_stack(spikes.detach(), targets)
                output.append(activations)
            output = torch.stack(output)
            if predict_class_over_time(output, steps) == targets.item():
                num_correct += 1
            samples_seen += 1

            progress_bar.set_description(f'Accuracy: {num_correct}/{samples_seen}  {num_correct / samples_seen}')


# cProfile.run("main()")

with torch.autograd.profiler.profile() as prof:
    main()

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=100))
