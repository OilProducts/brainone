import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils
import snntorch as snn
import snntorch.spikegen as spikegen

from tqdm import tqdm

import neurons

import cProfile, pstats

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

data_path = "~/robots/datasets/"
transform = transforms.Compose(
    [
        transforms.Resize((28, 28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
    ]
)


class STDPLinear(nn.Module):
    """A Linear layer that uses Spike-Timing Dependent Plasticity (STDP) for learning.

    Attributes:
    - in_features (int): Number of input features.
    - out_features (int): Number of output features.
    - batch_size (int): Batch size.
    - membrane_decay (float): Leak factor of the LIF neuron membrane.
    - threshold (float): Threshold for spike generation.
    - reset (float): Reset value for the membrane potential after spike generation.
    - a_pos (float): STDP parameter for potentiation.
    - a_neg (float): STDP parameter for depression.
    - trace_decay (float): Decay factor for the STDP traces. This quantity tracks the time difference between pre- and
                           post-synaptic spikes.
    """

    def __init__(self, in_features, out_features, batch_size=128, membrane_decay=.9, threshold_reset=1,
                 threshold_decay=.99, membrane_reset=0, a_pos=0.005, a_neg=0.005, trace_decay=.85,
                 plasticity_reward=1, plasticity_punish=1, device='cpu'):
        super().__init__()

        # Initializing weights, membrane potentials, and STDP-related variables
        self.weights = torch.rand(in_features, out_features, device=device) * .01
        self.membrane = torch.zeros(batch_size, out_features, device=device)
        self.thresholds = torch.ones(batch_size, out_features, device=device) * threshold_reset

        self.out_spikes = torch.zeros(batch_size, out_features, device=device)

        # Training parameters
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features

        # Neuron parameters
        self.membrane_reset = membrane_reset
        self.membrane_decay = membrane_decay
        self.threshold_reset = threshold_reset
        self.threshold_decay = threshold_decay
        self.threshold_targets = torch.full((batch_size, out_features), threshold_reset, dtype=torch.float ,device=device)

        # Plasticity parameters
        self.plasticity_reward = plasticity_reward
        self.plasticity_punish = plasticity_punish

        # STDP parameters
        self.a_pos = a_pos
        self.a_neg = a_neg

        # Initialize traces
        self.trace_pre = torch.zeros(in_features, device=device)
        self.trace_post = torch.zeros(out_features, device=device)
        self.trace_decay = trace_decay

    def forward(self, in_spikes, train=True):
        """Forward pass of the STDP Linear layer."""

        # Simulate the LIF neurons
        self.out_spikes, self.membrane, self.threshold = (
            neurons.LIF_with_threshold_decay(in_spikes,
                                             self.weights,
                                             self.membrane,
                                             self.membrane_decay,
                                             self.thresholds,
                                             self.threshold_reset,
                                             self.threshold_decay,
                                             self.membrane_reset))

        if train:
            # Update traces
            self.trace_pre = self.trace_pre * self.trace_decay + in_spikes
            self.trace_post = self.trace_post * self.trace_decay + self.out_spikes

            torch.clamp(self.trace_pre, 0, 1, out=self.trace_pre)
            torch.clamp(self.trace_post, 0, 1, out=self.trace_post)

            # Compute STDP weight changes using traces
            weight_changes = self.compute_stdp_with_trace(self.trace_pre, self.trace_post)

            # Save the last weight change for potential reward/punishment adjustments
            self.last_weight_change = weight_changes  # .clone()

            # Apply the STDP-induced weight changes
            avg_weight_changes = weight_changes.mean(dim=0)  # .t()
            self.weights += avg_weight_changes

            self.weights = torch.clamp(self.weights, -0.1, .5)

        return self.out_spikes

    # @torch.compile
    def compute_stdp_with_trace(self, trace_pre, trace_post):
        # This is a simplified STDP rule using traces, adjust as needed
        potentiation = trace_post.unsqueeze(1) * self.a_pos * trace_pre.unsqueeze(2)
        depression = trace_pre.unsqueeze(2) * self.a_neg * trace_post.unsqueeze(1)
        return potentiation - depression

    def reset_hidden_state(self):
        self.membrane = torch.zeros(self.batch_size, self.out_features, device=device)
        self.trace_pre = torch.zeros(self.in_features, device=device)
        self.trace_post = torch.zeros(self.out_features, device=device)

    def apply_reward(self, factor):
        """Modifies the last weight update based on a reward/punishment factor.

        Args:
        - factor (float): The factor by which to scale the last weight change.
                          A value > 1 indicates a reward, while a value < 1 indicates a punishment.
        """

        avg_last_weight_change = self.last_weight_change.mean(dim=0)
        self.weights += avg_last_weight_change * (factor - 1)


class SpikingNetwork(nn.Module):
    def __init__(self, batch_size=128, a_pos=.005, a_neg=.005, plasticity_reward=1, plasticity_punish=1, device='cuda'):
        super(SpikingNetwork, self).__init__()
        self.a_pos = a_pos
        self.a_neg = a_neg

        self.layer_1 = STDPLinear(784, 500,
                                  a_pos=self.a_pos,
                                  a_neg=self.a_neg,
                                  plasticity_reward=plasticity_reward,
                                  plasticity_punish=plasticity_punish,
                                  device=device,
                                  batch_size=batch_size)
        self.layer_2 = STDPLinear(500, 200,
                                  a_pos=self.a_pos,
                                  a_neg=self.a_neg,
                                  plasticity_reward=plasticity_reward,
                                  plasticity_punish=plasticity_punish,
                                  device=device,
                                  batch_size=batch_size)
        self.layer_3 = STDPLinear(200, 100,
                                  a_pos=self.a_pos,
                                  a_neg=self.a_neg,
                                  plasticity_reward=plasticity_reward,
                                  plasticity_punish=plasticity_punish,
                                  device=device,
                                  batch_size=batch_size)
        #
        # self.layer_2 = STDPLinear(600, 500,
        #                           a_pos=self.a_pos,
        #                           a_neg=self.a_neg,
        #                           plasticity_reward=plasticity_reward,
        #                           plasticity_punish=plasticity_punish,
        #                           device=device,
        #                           batch_size=batch_size)
        #
        # self.layer_3 = STDPLinear(600, 500,
        #                           a_pos=self.a_pos,
        #                           a_neg=self.a_neg,
        #                           plasticity_reward=plasticity_reward,
        #                           plasticity_punish=plasticity_punish,
        #                           device=device,
        #                           batch_size=batch_size)
        # self.layer_4 = STDPLinear(600, 200,
        #                           a_pos=self.a_pos,
        #                           a_neg=self.a_neg,
        #                           plasticity_reward=plasticity_reward,
        #                           plasticity_punish=plasticity_punish,
        #                           device=device,
        #                           batch_size=batch_size)
        # self.layer_5 = STDPLinear(200, 100,
        #                           a_pos=self.a_pos,
        #                           a_neg=self.a_neg,
        #                           plasticity_reward=plasticity_reward,
        #                           plasticity_punish=plasticity_punish,
        #                           device=device,
        #                           batch_size=batch_size)

    def forward(self, x, labels, train=True):
        layer_1_out = self.layer_1(x, train=train)
        layer_2_out = self.layer_2(layer_1_out, train=train)

        layer_3_out = self.layer_3(layer_2_out, train=train)

        # # print(f'total spikes in layer 1: {layer_1_out.sum()}')
        # layer_2_out = self.layer_2(torch.cat((layer_1_out, self.layer_5.out_spikes), 1), train=train)
        # # print(f'total spikes in layer 2: {layer_2_out.sum()}')
        # layer_3_out = self.layer_3(torch.cat((layer_2_out, self.layer_5.out_spikes), 1), train=train)
        # # print(f'total spikes in layer 3: {layer_3_out.sum()}')
        # layer_4_out = self.layer_4(torch.cat((layer_3_out, self.layer_5.out_spikes), 1), train=train)
        # # print(f'total spikes in layer 4: {layer_4_out.sum()}')
        # layer_5_out = self.layer_5(layer_4_out, train=train)
        # # print(f'total spikes in layer 5: {layer_5_out.sum()}')

        return layer_3_out

    def apply_reward(self, factor):
        self.layer_1.apply_reward(factor)
        self.layer_2.apply_reward(factor)
        self.layer_3.apply_reward(factor)
        # self.layer_4.apply_reward(factor)
        # self.layer_5.apply_reward(factor)

    def reset_hidden_state(self):
        self.layer_1.reset_hidden_state()
        self.layer_2.reset_hidden_state()
        self.layer_3.reset_hidden_state()
        # self.layer_4.reset_hidden_state()
        # self.layer_5.reset_hidden_state()


def pool_spikes(output_spikes):
    # print(f'Output spikes shape: {output_spikes.shape}')
    pooled_spikes = output_spikes.view(-1, 10, 10).sum(1)  # Assuming output_spikes has shape [100]
    # print(f'Pooled spikes shape: {pooled_spikes.shape}')
    return pooled_spikes


def validate(network, batch_size, num_steps, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.view(batch_size, -1)
            output_spike_accumulator = torch.zeros(batch_size, 100, device=device)

            for step in range(num_steps):
                in_spikes = spikegen.rate(inputs, 1).squeeze(0)
                output_spikes = network(in_spikes, train=False)
                output_spike_accumulator += output_spikes

            _, predicted_classes = pool_spikes(output_spike_accumulator).max(dim=1)
            # _, predicted_classes = output_spike_accumulator.max(dim=1)
            correct_predictions = (predicted_classes == labels).float()
            correct += correct_predictions.sum().item()  # Summing over the batch

            # _, predicted_class = output_spikes.sum(dim=0).max(dim=0)
            total += labels.size(0)
            network.reset_hidden_state()
            # correct += (predicted_class == labels).sum().item()
    return 100 * correct / total


def main():
    mnist_training_data = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )
    mnist_test_data = datasets.MNIST(
        data_path, train=False, download=True, transform=transform)

    # Training param
    num_epochs = 10
    num_steps = 100
    plasticity_reward = 1
    plasticity_punish = -.2
    batch_size = 16
    subset = 1

    mnist_training_data = utils.data_subset(mnist_training_data, subset)
    mnist_test_data = utils.data_subset(mnist_test_data, subset)

    # Initialize dataloaders
    train_loader = DataLoader(
        mnist_training_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=True,
        # prefetch_factor=8,
    )
    print("Train loader batches:", len(train_loader))

    test_loader = DataLoader(
        mnist_test_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    print("Test loader batches:", len(test_loader))

    # Initialize network
    network = SpikingNetwork(batch_size=batch_size,
                             a_pos=.002,
                             a_neg=.002,
                             plasticity_reward=plasticity_reward,
                             plasticity_punish=plasticity_punish,
                             device=device)

    for epoch in range(num_epochs):
        num_correct = 0  # Number of correct predictions
        samples_seen = 0  # Total number of samples processed

        correct_counts = torch.zeros(10, device=device)
        total_counts = torch.zeros(10, device=device)
        progress_bar = tqdm(iter(train_loader), total=len(train_loader), unit_scale=batch_size)
        for inputs, labels in progress_bar:
            # Move inputs and labels to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Convert inputs to spike trains
            inputs = inputs.view(batch_size, -1)  # This will reshape the tensor to [batch_size, 784]
            output_spike_accumulator = torch.zeros(batch_size, 100, device=device)

            # threshold = torch.tensor(batch_size, 100, .4, device=device)
            # threshold[labels] = .5

            # Adjust thresholds for the correct class neurons

            for idx, label in enumerate(labels):
                start_idx = label * 10  # 10 neurons_per_class
                end_idx = start_idx + 10  # 10 neurons_per_class

                network.layer_3.threshold_targets[idx, start_idx:end_idx] *= .5
                network.layer_3.thresholds = network.layer_3.threshold_targets

            for step in range(num_steps):
                in_spikes = spikegen.rate(inputs, 1).squeeze(0)

                # Forward pass through the network
                output_spikes = network(in_spikes, labels)

                # Accumulate spikes
                output_spike_accumulator += output_spikes

            network.layer_3.threshold_targets = torch.full((batch_size, 100), 1, dtype=torch.float, device=device)
            # Determine the predicted class based on the accumulated spikes
            _, predicted_classes = pool_spikes(output_spike_accumulator).max(dim=1)

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
        val_accuracy = validate(network, batch_size, num_steps, test_loader)
        print(f"Validation Accuracy after epoch {epoch + 1}: {val_accuracy:.2f}%")


with torch.no_grad():
    # main()

    # profiler = cProfile.Profile()
    # profiler.enable()
    main()
    # profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime').print_stats(10)
