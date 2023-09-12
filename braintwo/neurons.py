import torch


#@torch.compile
def LIFNeuron(input_spikes: torch.Tensor, weights: torch.Tensor, membrane: torch.Tensor, beta: float, threshold: float,
              reset: float):
    """Leaky Integrate-and-Fire (LIF) Neuron model.

    Args:
    - input_spikes (torch.Tensor): Input current to the neuron.
    - membrane (torch.Tensor): Membrane potential of the neuron.
    - beta (float): Leak factor of the neuron, determines the rate at which membrane potential decays.
    - threshold (float): Threshold for spike generation.
    - reset (float): Reset value for the membrane potential after spike generation.

    Returns:
    - spike (torch.Tensor): Binary tensor indicating whether a spike has been generated.
    - membrane (torch.Tensor): Updated membrane potential.
    """

    # Compute weighted input spikes
    weighted_input_spikes = torch.mm(input_spikes, weights)

    # Compute new membrane potential by adding input and applying leak factor
    membrane = membrane * beta + weighted_input_spikes

    # Generate spikes wherever membrane potential exceeds threshold
    spike = (membrane > threshold).float()

    # Reset the membrane potential wherever spikes are generated
    membrane = torch.where(spike.bool(), reset, membrane)

    return spike, membrane

#@torch.compile
def LIF_with_threshold_decay(input_spikes: torch.Tensor,
                             weights: torch.Tensor,
                             membrane: torch.Tensor,
                             beta: float,
                             thresholds: torch.Tensor,
                             threshold_targets: torch.Tensor,
                             threshold_decay: float,
                             reset: float):
    """Leaky Integrate-and-Fire (LIF) Neuron model with threshold decay.

    Args:
    - input_spikes (torch.Tensor): Input current to the neuron.
    - membrane (torch.Tensor): Membrane potential of the neuron.
    - beta (float): Leak factor of the neuron, determines the rate at which membrane potential decays.
    - thresholds (torch.Tensor): Threshold for spike generation.
    - threshold_reset (torch.Tensor): Value to reset the threshold to after spike generation.
    - threshold_decay (float): Factor by which to decay the threshold.
    - reset (float): Reset value for the membrane potential after spike generation.

    Returns:
    - spike (torch.Tensor): Binary tensor indicating whether a spike has been generated.
    - membrane (torch.Tensor): Updated membrane potential.
    - threshold (torch.Tensor): Updated threshold.
    """

    # Compute weighted input spikes
    weighted_input_spikes = torch.mm(input_spikes, weights)

    # Compute new membrane potential by adding input and applying leak factor
    membrane = membrane * beta + weighted_input_spikes

    # Generate spikes wherever membrane potential exceeds threshold
    spike = (membrane > thresholds).float()

    # Reset the membrane potential wherever spikes are generated
    membrane = torch.where(spike.bool(), reset, membrane)

    # Decay the threshold for all neurons
    thresholds = thresholds * threshold_decay

    # reset the threshold wherever spikes are generated
    thresholds = torch.where(spike.bool(), threshold_targets, thresholds)

    return spike, membrane, thresholds
