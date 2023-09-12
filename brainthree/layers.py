import jax
import jax.numpy as jnp
from flax import linen as nn

import neurons


def custom_weight_initializer(rng, shape):
    return jax.random.uniform(rng, shape) * (1. / shape[0])


class STDPLinear(nn.Module):
    in_features: int
    out_features: int
    batch_size: int
    threshold_reset: float = 1
    threshold_decay: float = .95
    membrane_reset: float = .1
    membrane_decay: float = .99
    a_pos: float = .005
    a_neg: float = .005
    trace_decay: float = .95
    plasticity_reward: float = 1
    plasticity_punish: float = 1

    def setup(self):
        # Initializing weights
        self.weights = self.param('weight', custom_weight_initializer, (self.in_features, self.out_features))
        self.membrane = jnp.ones((self.batch_size, self.out_features)) * self.membrane_reset
        self.thresholds = jnp.ones((self.batch_size, self.out_features)) * self.threshold_reset

        self.out_spikes = jnp.zeros((self.batch_size, self.out_features))

        self.trace_pre = jnp.ones(self.in_features)
        self.trace_post = jnp.ones(self.out_features)

    def __call__(self, in_spikes, membrane, thresholds):
        # Simulate the LIF neurons
        new_out_spikes, new_membrane, new_thresholds = neurons.LIF_with_threshold_decay(
            in_spikes, self.weights, self.membrane, self.membrane_decay,
            self.thresholds, self.threshold_reset, self.threshold_decay, self.membrane_reset
        )

        # Update traces
        self.trace_pre = self.trace_pre * self.trace_decay + in_spikes
        self.trace_post = self.trace_post * self.trace_decay + self.out_spikes

        self.trace_pre = jnp.clip(self.trace_pre, 0, 1)
        self.trace_post = jnp.clip(self.trace_post, 0, 1)

        # Compute STDP weight changes using traces
        weight_changes = self.compute_stdp_with_trace(self.trace_pre, self.trace_post)

        # Apply the STDP-induced weight changes
        avg_weight_changes = jnp.sum(weight_changes, axis=0)
        self.weights += avg_weight_changes

        self.weights = jnp.clip(self.weights, -0.1, 0.1)

        return new_out_spikes, new_membrane, new_thresholds

    def compute_stdp_with_trace(self, trace_pre, trace_post):
        # This is a simplified STDP rule using traces, adjust as needed
        potentiation = jnp.outer(trace_post, trace_pre) * self.a_pos
        depression = jnp.outer(trace_pre, trace_post) * self.a_neg
        return potentiation - depression

    def reset_hidden_state(self):
        self.membrane = jnp.zeros((self.batch_size, self.out_features))
        self.trace_pre = jnp.zeros(self.in_features)
        self.trace_post = jnp.zeros(self.out_features)

    def apply_reward(self, factor):
        avg_last_weight_change = jnp.sum(self.last_weight_change, axis=0)
        self.weights += avg_last_weight_change * (factor - 1)
