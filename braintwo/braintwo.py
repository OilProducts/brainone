import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from snntorch import utils
import snntorch as snn
import snntorch.spikegen as spikegen

from tqdm import tqdm

import cProfile

if torch.cuda.is_available():
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


# def LIFNeuron(input, membrane, beta, threshold, reset):
#     membrane = membrane * beta + input
#     spike = membrane > threshold
#     membrane = membrane - (spike * reset)
#     return spike, membrane


# def LIFNeuron(input, membrane, beta, threshold, reset):
#     membrane = membrane * beta + input
#     spike = membrane > threshold
#     membrane = torch.where(spike, membrane - reset, membrane)
#     return spike.float(), membrane


# def LIFNeuron(input, membrane, beta, threshold, reset):
#     membrane = membrane * beta + input
#     spike = (membrane > threshold).float()
#     membrane = torch.where(spike.bool(), membrane - reset, membrane)
#     return spike, membrane


def LIFNeuron(input, membrane, beta, threshold, reset):
    # print(f"Inside LIFNeuron - input shape: {input.shape}")  # Expected: [500]
    # print(f"Inside LIFNeuron - membrane shape: {membrane.shape}")  # Expected: [500]

    membrane = membrane * beta + input
    spike = (membrane > threshold).float()
    membrane = torch.where(spike.bool(), membrane - reset, membrane)

    # print(f"Inside LIFNeuron - output spike shape: {spike.shape}")  # Expected: [500]

    return spike, membrane


class STDPLinear(nn.Module):
    def __init__(self, in_features, out_features, batch_size=128, beta=.99, threshold=1, reset=.8, tau_pos=20.0, tau_neg=20.0,
                 a_pos=0.005, a_neg=0.005, device='cuda'):
        super().__init__()
        #self.weights = torch.rand(in_features, out_features)
        self.weights = nn.Linear(in_features, out_features, device=device)
        self.membrane = torch.zeros(batch_size, out_features, device=device)
        self.delta_pre = torch.zeros(batch_size, in_features, device=device)
        self.delta_fire = torch.zeros(batch_size, out_features, device=device)

        self.beta = beta
        self.threshold = threshold
        self.reset = reset
        self.plasticity_reward = 2
        self.plasticity_punish = .5

        # STDP parameters
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.a_pos = a_pos
        self.a_neg = a_neg

        # # Pre-compute exponentials for a range of delta_t values
        # self.max_delta = 100  # or whatever maximum delta_t you expect
        # self.exp_values_pos = torch.exp(-torch.arange(1, self.max_delta) / self.tau_pos)
        # self.exp_values_neg = torch.exp(torch.arange(1, self.max_delta) / self.tau_neg)


    def forward(self, in_spikes):
        # print(f'Initial in_spikes shape: {in_spikes.shape}')  # Should be [784]

        weighted_input = torch.mm(in_spikes, self.weights.weight.t())
        # print(f'Weighted input shape: {weighted_input.shape}')  # Should be [1, 500]
        #
        # print(f'weighted_input: {weighted_input.squeeze(0).shape}')
        # #weighted_input = self.weights(in_spikes)

        out_spikes, out_membrane = LIFNeuron(weighted_input.squeeze(), self.membrane, self.beta, self.threshold,
                                             self.reset)
        # print(f'Out spikes shape: {out_spikes.shape}')  # Should be [500]

        self.membrane = out_membrane

        # Update delta_pre and delta_fire
        self.delta_pre = self.delta_pre + 1
        self.delta_pre[in_spikes.bool()] = 0

        self.delta_fire = self.delta_fire + 1
        # print(f'out_spikes: {out_spikes.shape}')
        # print(f'delta_fire: {self.delta_fire.shape}')
        self.delta_fire[out_spikes.bool()] = 0

        # Compute delta_t for STDP
        delta_t = self.delta_fire.unsqueeze(1) - self.delta_pre.unsqueeze(2)
        # print(f'delta_t: {delta_t.shape}')

        # Compute STDP weight changes
        weight_changes = self.compute_stdp(delta_t)

        # Save the last weight change
        self.last_weight_change = weight_changes.clone()

        # Apply weight changes
        # print(f'weight_changes: {weight_changes.shape}')
        # print(f'weights: {self.weights.weight.shape}')
        avg_weight_changes = weight_changes.mean(dim=0).t()
        self.weights.weight.data += avg_weight_changes

        return out_spikes

    def compute_stdp(self, delta_t):
        potentiation = (delta_t > 0).float() * self.a_pos * torch.exp(-delta_t / self.tau_pos)
        depression = (delta_t < 0).float() * self.a_neg * torch.exp(delta_t / self.tau_neg)
        return potentiation - depression


    # def compute_stdp(self, delta_t):
    #     # Masks for potentiation and depression
    #     pot_mask = (delta_t > 0).float()
    #     dep_mask = (delta_t < 0).float()
    #
    #     # In-place exponential computation for potentiation and depression
    #     delta_t_pos = -delta_t * pot_mask / self.tau_pos
    #     delta_t_neg = delta_t * dep_mask / self.tau_neg
    #
    #     torch.exp_(delta_t_pos)
    #     torch.exp_(delta_t_neg)
    #
    #     potentiation = pot_mask * self.a_pos * delta_t_pos
    #     depression = dep_mask * self.a_neg * delta_t_neg
    #
    #     return potentiation - depression

    def apply_reward(self, factor):
        """
        Modifies the last weight update based on a reward/punishment factor.
        """
        avg_last_weight_change = self.last_weight_change.mean(dim=0).t()
        self.weights.weight.data += avg_last_weight_change * (factor - 1)


class SpikingNetwork(nn.Module):
    def __init__(self):
        super(SpikingNetwork, self).__init__()
        # self.layer_1 = STDPLinear(784, 500)
        # self.layer_2 = STDPLinear(500, 500)
        # self.layer_3 = STDPLinear(500, 500)
        # self.layer_4 = STDPLinear(500, 200)
        # self.layer_5 = STDPLinear(200, 10)

        self.layer_1 = STDPLinear(784, 500)
        self.layer_2 = STDPLinear(500, 200)
        self.layer_3 = STDPLinear(200, 10)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        # x = self.layer_4(x)
        # x = self.layer_5(x)
        return x

    def apply_reward(self, factor):
        self.layer_1.apply_reward(factor)
        self.layer_2.apply_reward(factor)
        self.layer_3.apply_reward(factor)


def main():
    mnist_training_data = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )
    mnist_test_data = datasets.MNIST(
        data_path, train=False, download=True, transform=transform)

    mnist_training_data = utils.data_subset(mnist_training_data, 1)
    mnist_test_data = utils.data_subset(mnist_test_data, 1)

    batch_size = 128

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
        num_workers=8
    )
    print("Test loader batches:", len(test_loader))

    # Initialize network
    network = SpikingNetwork()

    # Training param
    num_epochs = 10
    num_steps = 100
    plasticity_reward = 2
    plasticity_punish = 0.5

    for epoch in range(num_epochs):
        num_correct = 0  # Number of correct predictions
        samples_seen = 0  # Total number of samples processed

        progress_bar = tqdm(iter(train_loader), total=len(train_loader), unit_scale=batch_size)
        for inputs, labels in progress_bar:
            # Move inputs and labels to GPU
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            # Convert inputs to spike trains
            inputs = inputs.view(batch_size, -1)  # This will reshape the tensor to [batch_size, 784]
            output_spike_accumulator = torch.zeros(batch_size, 10, device='cuda')

            for step in range(num_steps):
                in_spikes = spikegen.rate(inputs, 1).squeeze(0)

                # Forward pass through the network
                output_spikes = network(in_spikes)

                # Accumulate spikes
                output_spike_accumulator += output_spikes

                # # Determine the predicted class based on the spikes in the last layer
                # _, predicted_class = output_spikes.sum(dim=0).max(dim=0)

            # Determine the predicted class based on the accumulated spikes
            _, predicted_classes = output_spike_accumulator.max(dim=1)

            correct_predictions = (predicted_classes == labels).float()
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
        torch.cuda.empty_cache()


with torch.no_grad():
    main()
    #cProfile.run("main()")
