import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils
import snntorch as snn
import snntorch.spikegen as spikegen

from tqdm import tqdm

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


def LIFNeuron(input, membrane, beta, threshold, reset):
    """Leaky Integrate-and-Fire (LIF) Neuron model.

    Args:
    - input (torch.Tensor): Input current to the neuron.
    - membrane (torch.Tensor): Membrane potential of the neuron.
    - beta (float): Leak factor of the neuron, determines the rate at which membrane potential decays.
    - threshold (float): Threshold for spike generation.
    - reset (float): Reset value for the membrane potential after spike generation.

    Returns:
    - spike (torch.Tensor): Binary tensor indicating whether a spike has been generated.
    - membrane (torch.Tensor): Updated membrane potential.
    """

    # Compute new membrane potential by adding input and applying leak factor
    membrane = membrane * beta + input

    # Generate spikes wherever membrane potential exceeds threshold
    spike = (membrane > threshold).float()

    # Reset the membrane potential wherever spikes are generated
    membrane = torch.where(spike.bool(), membrane - reset, membrane)

    return spike, membrane


class STDPLinear(nn.Module):
    """A Linear layer that uses Spike-Timing Dependent Plasticity (STDP) for learning.

    Attributes:
    - weights (nn.Linear): The learnable weights of the module.
    - membrane (torch.Tensor): Membrane potentials for the neurons.
    - delta_pre (torch.Tensor): Time since last spike for input neurons.
    - delta_fire (torch.Tensor): Time since last spike for output neurons.
    - STDP parameters (floats): Parameters that determine the behavior of STDP (e.g., tau_pos, tau_neg, etc.)
    """

    def __init__(self, in_features, out_features, batch_size=128, beta=.99, threshold=1, reset=.8,
                 tau_pos=20.0, tau_neg=20.0, a_pos=0.005, a_neg=0.005, trace_decay=.9, device='cpu'):
        super().__init__()

        # Initializing weights, membrane potentials, and STDP-related variables
        self.weights = nn.Linear(in_features, out_features, device=device)
        self.membrane = torch.zeros(batch_size, out_features, device=device)

        # Training parameters
        self.batch_size = batch_size
        self.in_features = in_features
        self.out_features = out_features

        # Neuron parameters
        self.beta = beta
        self.threshold = threshold
        self.reset = reset

        # Plasticity parameters
        self.plasticity_reward = 2
        self.plasticity_punish = .5

        # STDP parameters
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.a_pos = a_pos
        self.a_neg = a_neg

        # Initialize traces
        self.trace_pre = torch.zeros(in_features, device=device)
        self.trace_post = torch.zeros(out_features, device=device)
        self.trace_decay = trace_decay

    def forward(self, in_spikes, train=True):
        """Forward pass of the STDP Linear layer."""

        # Compute the input for the LIF neurons
        weighted_input = torch.mm(in_spikes, self.weights.weight.t())

        # Simulate the LIF neurons
        out_spikes, out_membrane = LIFNeuron(weighted_input.squeeze(), self.membrane,
                                             self.beta, self.threshold, self.reset)
        self.membrane = out_membrane

        # # Update the time since the last spike for each neuron
        # self.delta_pre = self.delta_pre + 1
        # self.delta_pre[in_spikes.bool()] = 0
        # self.delta_fire = self.delta_fire + 1
        # self.delta_fire[out_spikes.bool()] = 0

        # Update traces
        self.trace_pre = self.trace_pre * self.trace_decay + in_spikes
        self.trace_post = self.trace_post * self.trace_decay + out_spikes

        # Compute STDP weight changes using traces
        weight_changes = self.compute_stdp_with_trace(self.trace_pre, self.trace_post)

        # Save the last weight change for potential reward/punishment adjustments
        self.last_weight_change = weight_changes  # .clone()

        # Apply the STDP-induced weight changes
        avg_weight_changes = weight_changes.sum(dim=0).t()
        self.weights.weight.data += avg_weight_changes

        self.weights.weight.data = torch.clamp(self.weights.weight.data, -.1, .1)

        return out_spikes

    @torch.compile
    def compute_stdp_with_trace(self, trace_pre, trace_post):
        # This is a simplified STDP rule using traces, adjust as needed
        potentiation = trace_post.unsqueeze(1) * self.a_pos * trace_pre.unsqueeze(2)
        depression = trace_pre.unsqueeze(2) * self.a_neg * trace_post.unsqueeze(1)
        return potentiation - depression

    def reset(self):
        self.membrane = torch.zeros(self.batch_size, self.out_features, device=device)
        self.trace_pre = torch.zeros(self.in_features, device=device)
        self.trace_post = torch.zeros(self.out_features, device=device)

    @torch.compile
    def compute_stdp(self, delta_t):
        """Compute the weight changes due to STDP rules.

        Args:
        - delta_t (torch.Tensor): The difference in spike timings between pre and post-synaptic neurons.

        Returns:
        - torch.Tensor: The computed weight changes.
        """

        # Potentiation (strengthening) due to pre-before-post spikes
        potentiation = (delta_t > 0) * self.a_pos * torch.exp(-delta_t / self.tau_pos)

        # Depression (weakening) due to post-before-pre spikes
        depression = (delta_t < 0) * self.a_neg * torch.exp(delta_t / self.tau_neg)

        return potentiation - depression

    def apply_reward(self, factor):
        """Modifies the last weight update based on a reward/punishment factor.

        Args:
        - factor (float): The factor by which to scale the last weight change.
                          A value > 1 indicates a reward, while a value < 1 indicates a punishment.
        """

        avg_last_weight_change = self.last_weight_change.mean(dim=0).t()
        self.weights.weight.data += avg_last_weight_change * (factor - 1)


class SpikingNetwork(nn.Module):
    def __init__(self, batch_size=128, learning_rate=.01, device='cuda'):
        super(SpikingNetwork, self).__init__()
        # self.layer_1 = STDPLinear(784, 500)
        # self.layer_2 = STDPLinear(500, 500)
        # self.layer_3 = STDPLinear(500, 500)
        # self.layer_4 = STDPLinear(500, 200)
        # self.layer_5 = STDPLinear(200, 10)

        self.learning_rate = learning_rate

        self.layer_1 = STDPLinear(784, 500,
                                  a_pos=self.learning_rate,
                                  a_neg=self.learning_rate,
                                  device=device,
                                  batch_size=batch_size)
        self.layer_2 = STDPLinear(500, 200,
                                  a_pos=self.learning_rate,
                                  a_neg=self.learning_rate,
                                  device=device,
                                  batch_size=batch_size)
        self.layer_3 = STDPLinear(200, 10,
                                  a_pos=self.learning_rate,
                                  a_neg=self.learning_rate,
                                  device=device,
                                  batch_size=batch_size)

    def forward(self, x, train=True):
        x = self.layer_1(x, train=train)
        x = self.layer_2(x, train=train)
        x = self.layer_3(x, train=train)
        # x = self.layer_4(x)
        # x = self.layer_5(x)
        return x

    def apply_reward(self, factor):
        self.layer_1.apply_reward(factor)
        self.layer_2.apply_reward(factor)
        self.layer_3.apply_reward(factor)

    def reset(self):
        self.layer_1.reset()
        self.layer_2.reset()
        self.layer_3.reset()


def validate(network, batch_size, num_steps, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            inputs = inputs.view(batch_size, -1)
            output_spike_accumulator = torch.zeros(batch_size, 10, device=device)

            for step in range(num_steps):
                in_spikes = spikegen.rate(inputs, 1).squeeze(0)
                output_spikes = network(in_spikes)
                output_spike_accumulator += output_spikes

            _, predicted_classes = output_spike_accumulator.max(dim=1)
            correct_predictions = (predicted_classes == labels).float()
            correct += correct_predictions.sum().item()  # Summing over the batch

            # _, predicted_class = output_spikes.sum(dim=0).max(dim=0)
            total += labels.size(0)
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
    num_steps = 30
    plasticity_reward = 2
    plasticity_punish = 0.5
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
        num_workers=8,
        pin_memory=True,
        # prefetch_factor=8,
    )
    print("Train loader batches:", len(train_loader))

    test_loader = DataLoader(
        mnist_test_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=8
    )
    print("Test loader batches:", len(test_loader))

    # Initialize network
    network = SpikingNetwork(batch_size=batch_size,
                             learning_rate=.0001,
                             device=device)

    for epoch in range(num_epochs):
        num_correct = 0  # Number of correct predictions
        samples_seen = 0  # Total number of samples processed

        progress_bar = tqdm(iter(train_loader), total=len(train_loader), unit_scale=batch_size)
        for inputs, labels in progress_bar:
            # Move inputs and labels to GPU
            inputs = inputs.to(device)
            labels = labels.to(device)
            # Convert inputs to spike trains
            inputs = inputs.view(batch_size, -1)  # This will reshape the tensor to [batch_size, 784]
            output_spike_accumulator = torch.zeros(batch_size, 10, device=device)

            for step in range(num_steps):
                in_spikes = spikegen.rate(inputs, 1).squeeze(0)

                # Forward pass through the network
                output_spikes = network(in_spikes)

                # Accumulate spikes
                output_spike_accumulator += output_spikes

                # # Determine the predicted class based on the spikes in the last layer
                # _, predicted_class = output_spikes.sum(dim=0).max(dim=0)

            # Determine the predicted class based on the accumulated spikes
            # print(f'output_spike_accumulator: {output_spike_accumulator}')
            _, predicted_classes = output_spike_accumulator.max(dim=1)

            correct_predictions = (predicted_classes == labels).float()
            # print(f'correct_predictions: {correct_predictions}')
            num_correct += correct_predictions.sum().item()  # Summing over the batch
            reward_factors = correct_predictions * plasticity_reward + (1 - correct_predictions) * plasticity_punish
            for factor in reward_factors:
                network.apply_reward(factor)

            # Update statistics
            samples_seen += batch_size
            # if predicted_class == labels:
            #     num_correct += 1
            #     network.apply_reward(plasticity_reward)
            # else:
            #     network.apply_reward(plasticity_punish)

            # Update progress bar description
            accuracy = num_correct / samples_seen * 100
            progress_bar.set_description(
                f'Epoch: {epoch + 1}/{num_epochs} Accuracy: {accuracy:.2f}% ({num_correct}/{samples_seen})')

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
