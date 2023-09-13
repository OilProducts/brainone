import jax
import jax.numpy as jnp


def LIF_with_threshold_decay(input_spikes,
                             weights,
                             membrane,
                             beta,
                             thresholds,
                             threshold_targets,
                             threshold_decay,
                             membrane_reset):
    # Compute new membrane potential by adding input and applying leak factor
    membrane = membrane * beta + jnp.dot(input_spikes, weights)

    # Generate spikes wherever membrane potential exceeds threshold
    spike = (membrane > thresholds).astype(jnp.float32)

    # Reset the membrane potential and threshold wherever spikes are generated
    membrane = jnp.where(spike, membrane_reset, membrane)
    thresholds = jnp.where(spike, threshold_targets, thresholds * threshold_decay)

    return spike, membrane, thresholds