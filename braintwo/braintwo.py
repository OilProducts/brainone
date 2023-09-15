import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils
import snntorch as snn
import snntorch.spikegen as spikegen

from tqdm import tqdm

import neurons
import layers
import spike_utils

import cProfile
import pstats

torch.set_float32_matmul_precision('high')

torch.set_printoptions(threshold=100000)

if torch.cuda.is_available():
    print(f'Cuda compute version: {torch.cuda.get_device_capability(0)}')
    if torch.cuda.get_device_capability(0)[0] < 7:
        print("GPU compute capability is too low, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        print("GPU is available and being used")
else:
    device = torch.device("cpu")
    print("GPU is not available, using CPU instead")

#device = torch.device("cpu")

data_path = "~/robots/datasets/"
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)


class SimpleSpikeNetwork(nn.Module):
    def __init__(self, batch_size=128, a_pos=.005, a_neg=.005, plasticity_reward=1, plasticity_punish=1, device='cuda'):
        super(SimpleSpikeNetwork, self).__init__()
        self.a_pos = a_pos
        self.a_neg = a_neg

        self.layer_1 = layers.STDPLinear(784, 500,
                                         a_pos=self.a_pos,
                                         a_neg=self.a_neg,
                                         plasticity_reward=plasticity_reward,
                                         plasticity_punish=plasticity_punish,
                                         device=device,
                                         batch_size=batch_size)
        self.layer_2 = layers.STDPLinear(500, 200,
                                         a_pos=self.a_pos,
                                         a_neg=self.a_neg,
                                         plasticity_reward=plasticity_reward,
                                         plasticity_punish=plasticity_punish,
                                         device=device,
                                         batch_size=batch_size)
        self.layer_3 = layers.STDPLinear(200, 10,
                                         a_pos=self.a_pos,
                                         a_neg=self.a_neg,
                                         plasticity_reward=plasticity_reward,
                                         plasticity_punish=plasticity_punish,
                                         device=device,
                                         batch_size=batch_size)

    def forward(self, x, labels, train=True):
        layer_1_out = self.layer_1(x, train=train)
        layer_2_out = self.layer_2(layer_1_out, train=train)
        layer_3_out = self.layer_3(layer_2_out, train=train)
        # print(f'layer_2 threshold targets: {self.layer_2.threshold_targets}')

        return layer_3_out

    def apply_reward(self, factor):
        self.layer_1.apply_reward(factor)
        self.layer_2.apply_reward(factor)
        self.layer_3.apply_reward(factor)

    def reset_hidden_state(self):
        self.layer_1.reset_hidden_state()
        self.layer_2.reset_hidden_state()
        self.layer_3.reset_hidden_state()


class SpikingNetwork(nn.Module):
    def __init__(self, batch_size=128, a_pos=.005, a_neg=.005, plasticity_reward=1, plasticity_punish=1, device='cuda'):
        super(SpikingNetwork, self).__init__()
        self.a_pos = a_pos
        self.a_neg = a_neg

        self.layer_1 = layers.STDPLinear(784, 500,
                                  a_pos=self.a_pos,
                                  a_neg=self.a_neg,
                                  plasticity_reward=plasticity_reward,
                                  plasticity_punish=plasticity_punish,
                                  device=device,
                                  batch_size=batch_size)

        self.layer_2 = layers.STDPLinear(500, 500,
                                  a_pos=self.a_pos,
                                  a_neg=self.a_neg,
                                  plasticity_reward=plasticity_reward,
                                  plasticity_punish=plasticity_punish,
                                  device=device,
                                  batch_size=batch_size)

        self.layer_3 = layers.STDPLinear(500, 500,
                                  a_pos=self.a_pos,
                                  a_neg=self.a_neg,
                                  plasticity_reward=plasticity_reward,
                                  plasticity_punish=plasticity_punish,
                                  device=device,
                                  batch_size=batch_size)
        self.layer_4 = layers.STDPLinear(500, 500,
                                  a_pos=self.a_pos,
                                  a_neg=self.a_neg,
                                  plasticity_reward=plasticity_reward,
                                  plasticity_punish=plasticity_punish,
                                  device=device,
                                  batch_size=batch_size)
        self.layer_5 = layers.STDPLinear(500, 500,
                                  a_pos=self.a_pos,
                                  a_neg=self.a_neg,
                                  plasticity_reward=plasticity_reward,
                                  plasticity_punish=plasticity_punish,
                                  device=device,
                                  batch_size=batch_size)
        self.layer_6 = layers.STDPLinear(500, 500,
                                  a_pos=self.a_pos,
                                  a_neg=self.a_neg,
                                  plasticity_reward=plasticity_reward,
                                  plasticity_punish=plasticity_punish,
                                  device=device,
                                  batch_size=batch_size)
        self.layer_7 = layers.STDPLinear(500, 100,
                                  a_pos=self.a_pos,
                                  a_neg=self.a_neg,
                                  plasticity_reward=plasticity_reward,
                                  plasticity_punish=plasticity_punish,
                                  device=device,
                                  batch_size=batch_size)



    def forward(self, x, labels, train=True):
        layer_1_out = self.layer_1(x, train=train)
        layer_2_out = self.layer_2(layer_1_out, train=train)

        layer_3_out = self.layer_3(layer_2_out, train=train)
        layer_4_out = self.layer_4(layer_3_out, train=train)
        layer_5_out = self.layer_5(layer_4_out, train=train)
        layer_6_out = self.layer_6(layer_5_out, train=train)
        layer_7_out = self.layer_7(layer_6_out, train=train)

        return layer_7_out

    def apply_reward(self, factor):
        self.layer_1.apply_reward(factor)
        self.layer_2.apply_reward(factor)
        self.layer_3.apply_reward(factor)
        self.layer_4.apply_reward(factor)
        self.layer_5.apply_reward(factor)
        self.layer_6.apply_reward(factor)
        self.layer_7.apply_reward(factor)

    def reset_hidden_state(self):
        self.layer_1.reset_hidden_state()
        self.layer_2.reset_hidden_state()
        self.layer_3.reset_hidden_state()
        self.layer_4.reset_hidden_state()
        self.layer_5.reset_hidden_state()
        self.layer_6.reset_hidden_state()
        self.layer_7.reset_hidden_state()


def pool_spikes(output_spikes):
    # print(f'Output spikes: {output_spikes}')
    # print(f'Output spikes shape: {output_spikes.shape}')
    pooled_spikes = output_spikes.view(-1, 10, 10).sum(1)  # Assuming output_spikes has shape [100]
    # print(f'Pooled spikes: {pooled_spikes}')
    # print(f'Pooled spikes shape: {pooled_spikes.shape}')
    return pooled_spikes


def validate(network, batch_size, num_steps, test_loader):
    total_correct = 0
    total = 0

    correct_counts = torch.zeros(10, device=device)
    total_counts = torch.zeros(10, device=device)

    with torch.no_grad():
        progress_bar = tqdm(iter(test_loader), total=len(test_loader), unit_scale=batch_size)

        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.view(batch_size, -1)
            output_spike_accumulator = torch.zeros(batch_size, 100, device=device)

            for step in range(num_steps):
                in_spikes = spikegen.rate(inputs, 1).squeeze(0)
                output_spikes = network(in_spikes, labels, train=False)
                output_spike_accumulator += output_spikes

            _, predicted_classes = output_spike_accumulator.max(dim=1)
            # _, predicted_classes = output_spike_accumulator.max(dim=1)
            correct_predictions = (predicted_classes == labels).float()

            for idx, correct in enumerate(correct_predictions):
                if correct == 1.0:
                    correct_counts[labels[idx]] += 1
                total_counts[labels[idx]] += 1

            label_strings = []
            for label in range(10):
                accuracy_label = (correct_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0
                label_string = f"{label}: {int(correct_counts[label])}/{int(total_counts[label])}"  # ({accuracy_label:.2f}%)"
                label_strings.append(label_string)
            total_correct += correct_predictions.sum().item()  # Summing over the batch

            # _, predicted_class = output_spikes.sum(dim=0).max(dim=0)
            total += batch_size

            # Update progress bar description
            accuracy = total_correct / total * 100
            desc = f'Test Accuracy: {accuracy:.2f}% ({total_correct}/{total}) ' + ' '.join(
                label_strings)

            progress_bar.set_description(desc)

            network.reset_hidden_state()
            # correct += (predicted_class == labels).sum().item()
    return 100 * total_correct / total


def main():
    # Training param
    num_epochs = 20
    num_steps = 2000
    plasticity_reward = 1
    plasticity_punish = 1
    batch_size = 128
    shrink_factor = 1

    mnist_training_loader, mnist_test_loader = (
        spike_utils.get_mnist_dataloaders(shrink_factor=shrink_factor,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=0))

    network = SpikingNetwork(batch_size=batch_size,
                                a_pos=.001,
                                a_neg=.0005,
                                plasticity_reward=plasticity_reward,
                                plasticity_punish=plasticity_punish,
                                device=device)

    for epoch in range(num_epochs):
        num_correct = 0  # Number of correct predictions
        samples_seen = 0  # Total number of samples processed

        correct_counts = torch.zeros(10, device=device)
        total_counts = torch.zeros(10, device=device)
        progress_bar = tqdm(iter(mnist_training_loader), total=len(mnist_training_loader), unit_scale=batch_size)
        for inputs, labels in progress_bar:
            # Move inputs and labels to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Convert inputs to spike trains
            inputs = inputs.view(batch_size, -1)  # This will reshape the tensor to [batch_size, 784]
            output_spike_accumulator = torch.zeros(batch_size, 100, device=device)

            # Set label threshold to be lower
            # print(f'layer_2 threshold targets: {network.layer_2.threshold_targets}')
            # print(f'layer_2 threshold targets shape: {network.layer_2.threshold_targets.shape}')
            for idx, label in enumerate(labels):
                # print(f'idx: {idx}, label: {label}')
                # network.layer_3.threshold_targets[idx, label] = network.layer_3.threshold_reset * .7
                # network.layer_3.thresholds = network.layer_3.threshold_targets
                start_idx = label * 10  # 10 neurons_per_class
                end_idx = start_idx + 10  # 10 neurons_per_class
                #
                network.layer_7.threshold_targets[idx, start_idx:end_idx] *= .8
                network.layer_7.thresholds = network.layer_7.threshold_targets
            # print(f'layer_2 threshold targets: {network.layer_2.threshold_targets}')

            for step in range(num_steps):
                in_spikes = spikegen.rate(inputs, 1).squeeze(0)

                # Forward pass through the network
                output_spikes = network(in_spikes, labels)
                # print(f'output_spikes: {output_spikes}')

                # Accumulate spikes
                output_spike_accumulator += output_spikes

            # network.layer_3.threshold_targets = torch.full((batch_size, 10), 1, dtype=torch.float, device=device)

            network.layer_7.threshold_targets = torch.full((batch_size, 100), 1, dtype=torch.float, device=device)
            # Determine the predicted class based on the accumulated spikes
            # print(f'output_spike_accumulator: {output_spike_accumulator}')
            #_, predicted_classes = output_spike_accumulator.max(dim=1)
            _, predicted_classes = pool_spikes(output_spike_accumulator).max(dim=1) # TODO: make sure pool_spikes works correctly

            correct_predictions = (predicted_classes == labels).float()
            for idx, correct in enumerate(correct_predictions):
                if correct == 1.0:
                    correct_counts[labels[idx]] += 1
                total_counts[labels[idx]] += 1

            # Construct the string for the tqdm description:
            label_strings = []
            for label in range(10):
                accuracy_label = (correct_counts[label] / total_counts[label] * 100) if total_counts[label] > 0 else 0
                label_string = f"{label}: {int(correct_counts[label])}/{int(total_counts[label])}"  # ({accuracy_label:.2f}%)"
                label_strings.append(label_string)

            num_correct += correct_predictions.sum().item()  # Summing over the batch
            reward_factors = correct_predictions * plasticity_reward + (1 - correct_predictions) * plasticity_punish
            for factor in reward_factors:
                network.apply_reward(factor)

            # Update statistics
            samples_seen += batch_size

            # Update progress bar description
            accuracy = num_correct / samples_seen * 100
            desc = f'Epoch: {epoch + 1}/{num_epochs} Accuracy: {accuracy:.2f}% ({num_correct}/{samples_seen}) ' + ' '.join(
                label_strings)

            progress_bar.set_description(desc)
            network.reset_hidden_state()

        # After training for one epoch, validate the model
        val_accuracy = validate(network, batch_size, num_steps, mnist_test_loader)
        # print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")


with torch.no_grad():
    # main()

    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime').print_stats(10)
