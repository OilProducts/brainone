import torch
import torch.nn as nn

import neurons


class STDPLinear(nn.Module):
    """A Linear layer that uses Spike-Timing Dependent Plasticity (STDP) for
    learning.

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

    def __init__(self, in_features, out_features,
                 batch_size=128,
                 threshold_reset=1,
                 threshold_decay=.95,
                 membrane_reset=.1,
                 membrane_decay=.99,
                 a_pos=0.005, a_neg=0.005,
                 trace_decay=.95,
                 plasticity_reward=1,
                 plasticity_punish=1,
                 device='cpu'):
        super().__init__()

        # Initializing weights, membrane potentials, and STDP-related variables
        self.weights = torch.rand(in_features, out_features, device=device) * (1 / (in_features * 1))
        self.membrane = torch.ones(batch_size, out_features, device=device) * membrane_reset
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
        self.threshold_targets = torch.full((batch_size, out_features), threshold_reset, dtype=torch.float,
                                            device=device)

        # Plasticity parameters
        self.plasticity_reward = plasticity_reward
        self.plasticity_punish = plasticity_punish

        # STDP parameters
        self.a_pos = a_pos  # this is the learning rate for potentiation
        self.a_neg = a_neg  # this is the learning rate for depression

        # Initialize traces
        self.trace_pre = torch.ones(in_features, device=device)
        self.trace_post = torch.ones(out_features, device=device)
        self.trace_decay = trace_decay

        self.device = device

    #@torch.compile
    def forward(self, in_spikes, train=True):
        """Forward pass of the STDP Linear layer."""

        # Simulate the LIF neurons
        self.out_spikes, self.membrane, self.thresholds = (
            neurons.LIF_with_threshold_decay(in_spikes,
                                             self.weights,
                                             self.membrane,
                                             self.membrane_decay,
                                             self.thresholds,
                                             self.threshold_targets,
                                             self.threshold_decay,
                                             self.membrane_reset))

        # print(f'num in_spikes: {in_spikes.sum()}')
        # print(f'num out_spikes: {self.out_spikes.sum()}')

        if train:
            # Update traces
            self.trace_pre = self.trace_pre * self.trace_decay + in_spikes
            self.trace_post = self.trace_post * self.trace_decay + self.out_spikes

            torch.clamp(self.trace_pre, 0, 1, out=self.trace_pre)
            torch.clamp(self.trace_post, 0, 1, out=self.trace_post)

            # Compute STDP weight changes using traces
            weight_changes = self.compute_stdp_with_trace(self.trace_pre, self.trace_post)
            # print(f'in_spikes: {in_spikes.sum()}')
            #print(f'Weight changes: {weight_changes.abs().sum()/(self.in_features*self.out_features)}')
            # Save the last weight change for potential reward/punishment adjustments
            self.last_weight_change = weight_changes  # .clone()

            # Apply the STDP-induced weight changes
            avg_weight_changes = weight_changes.sum(dim=0)  # .t()
            self.weights += avg_weight_changes

            self.weights = torch.clamp(self.weights, -0.1, .1)

        return self.out_spikes

    #@torch.compile
    def compute_stdp_with_trace(self, trace_pre, trace_post):
        # This is a simplified STDP rule using traces, adjust as needed
        potentiation = trace_post.unsqueeze(1) * self.a_pos * trace_pre.unsqueeze(2)
        depression = trace_pre.unsqueeze(2) * self.a_neg * trace_post.unsqueeze(1)
        return potentiation - depression

    def reset_hidden_state(self):
        self.membrane = torch.ones(self.batch_size, self.out_features, device=self.device) * self.membrane_reset
        self.trace_pre = torch.ones(self.in_features, device=self.device)
        self.trace_post = torch.ones(self.out_features, device=self.device)

    def apply_reward(self, factor):
        """Modifies the last weight update based on a reward/punishment factor.

        Args:
        - factor (float): The factor by which to scale the last weight change.
                          A value > 1 indicates a reward, while a value < 1 indicates a punishment.
        """

        avg_last_weight_change = self.last_weight_change.sum(dim=0)
        self.weights += avg_last_weight_change * (factor - 1)
