from functools import partial
from typing import Union, Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import vjp, flatten_util
from jax.tree_util import tree_flatten
import flax.linen as nn
from flax import nnx

from .color import hsv2rgb

cache = lambda x: x
identity = lambda x: x
cos = jnp.cos
sin = jnp.sin
tanh = jnp.tanh
sigmoid = lambda x: jax.nn.sigmoid(x) * 2. - 1.
gaussian = lambda x: jnp.exp(-x**2) * 2. - 1.
relu = jax.nn.relu
activation_fn_map = dict(cache=cache, identity=identity, cos=cos, sin=sin, tanh=tanh, sigmoid=sigmoid, gaussian=gaussian, relu=relu)

class CPPN(nnx.Module):
    """
    CPPN Flax Model using NNX.
    Possible activations: cache (identity), identity, cos, sin, tanh, sigmoid, gaussian, relu.

    arch: str should be in the form "12;cache:15,gaussian:4,identity:2,sin:1" which means 12 layers, with each layer containing 15 neurons using cache, 4 neurons using gaussian, 2 neurons using identity, 1 neuron using sin
    inputs_str: str should be in the form "y,x,d,b" which means the inputs are y, x, d, b. Don't change this.
    init_scale_str: str should be in the form "default" or float. If default uses the default flax initialization scheme (lecun init). If float, it is the scale of the initialization variance (see code).
    """
    def __init__(self, arch: str, inputs_str: str, init_scale_str: str, *, rngs: nnx.Rngs):
        self.arch_str = arch
        self.inputs_str = inputs_str
        self.init_scale_str = init_scale_str

        n_layers_str, activation_neurons_str = self.arch_str.split(";")
        self.n_layers = int(n_layers_str)
        self.activations = [i.split(":")[0] for i in activation_neurons_str.split(",")]
        self.d_hidden = [int(i.split(":")[-1]) for i in activation_neurons_str.split(",")]
        self.dh_cumsum = list(np.cumsum(self.d_hidden))

        input_dim = len(self.inputs_str.split(","))

        self.hidden_layers = []
        current_in_features = input_dim
        for _ in range(self.n_layers):
            if self.init_scale_str == "default":
                layer = nnx.Linear(in_features=current_in_features, out_features=sum(self.d_hidden), use_bias=False, rngs=rngs)
            else:
                kernel_init = nnx.initializers.variance_scaling(scale=float(self.init_scale_str), mode="fan_in", distribution="truncated_normal")
                layer = nnx.Linear(in_features=current_in_features, out_features=sum(self.d_hidden), use_bias=False, kernel_init=kernel_init, rngs=rngs)
            self.hidden_layers.append(layer)
            current_in_features = sum(self.d_hidden)

        if self.init_scale_str == "default":
            self.output_layer = nnx.Linear(in_features=sum(self.d_hidden), out_features=3, use_bias=False, rngs=rngs)
        else:
            kernel_init = nnx.initializers.variance_scaling(scale=float(self.init_scale_str), mode="fan_in", distribution="truncated_normal")
            self.output_layer = nnx.Linear(in_features=sum(self.d_hidden), out_features=3, use_bias=False, kernel_init=kernel_init, rngs=rngs)

    def __call__(self, x):
        features = [x]
        for i_layer in range(self.n_layers):
            x = self.hidden_layers[i_layer](x)
            x = jnp.split(x, self.dh_cumsum[:-1]) # dh_cumsum includes total sum at the end, split needs section boundaries
            x = [activation_fn_map[activation](xi) for xi, activation in zip(x, self.activations)]
            x = jnp.concatenate(x)
            features.append(x)

        x = self.output_layer(x)
        features.append(x)
        h, s, v = x
        return (h, s, v), features

    def generate_image(self, img_size=256, return_features=False):
        """
        Generate an image from the CPPN at the resolution specified by img_size.
        Generate an image from the CPPN at the resolution specified by img_size.
        If return_features is True, return the intermediate activations of the CPPN as well.
        """
        inputs_dict = {}
        x_coords = y_coords = jnp.linspace(-1, 1, img_size)
        inputs_dict['x'], inputs_dict['y'] = jnp.meshgrid(x_coords, y_coords, indexing='ij')
        inputs_dict['d'] = jnp.sqrt(inputs_dict['x']**2 + inputs_dict['y']**2) * 1.4
        inputs_dict['b'] = jnp.ones_like(inputs_dict['x'])
        inputs_dict['xabs'], inputs_dict['yabs'] = jnp.abs(inputs_dict['x']), jnp.abs(inputs_dict['y'])

        # Prepare inputs based on self.inputs_str
        input_values = [inputs_dict[input_name] for input_name in self.inputs_str.split(",")]
        inputs_array = jnp.stack(input_values, axis=-1)

        # Apply the model (self) using jax.vmap
        (h, s, v), features = jax.vmap(jax.vmap(self))(inputs_array)

        r, g, b = hsv2rgb((h+1)%1, s.clip(0,1), jnp.abs(v).clip(0, 1))
        rgb = jnp.stack([r, g, b], axis=-1)
        if return_features:
            return rgb, features
        else:
            return rgb


class FlattenCPPNParameters():
    """
    Flatten the parameters of the CPPN to a single vector.
    Simplifies and makes useful for various things, like analysis.
    """
    def __init__(self, cppn_module_class, arch_str, inputs_str, init_scale_str):
        self.cppn_module_class = cppn_module_class
        self.arch_str = arch_str
        self.inputs_str = inputs_str
        self.init_scale_str = init_scale_str

        # For NNX, initialization requires actual RNGs.
        # We need a way to get parameter shapes without full initialization if possible,
        # or perform a dummy initialization to get shapes.
        # NNX modules are stateful, so 'params' are part of the module state.
        # Flattening might involve serializing the state.

        # Placeholder: Actual flattening/reshaping for NNX needs careful consideration
        # of how state is managed and serialized. This part will likely need significant
        # changes to align with NNX's stateful nature.
        # For now, let's assume we can get a representative state for shape inference.

        rng = nnx.Rngs(params=jax.random.key(0)) # NNX uses specific RNG streams
        d_in = len(self.inputs_str.split(","))
        # Initialize a dummy CPPN to get its state structure
        dummy_cppn = self.cppn_module_class(arch=self.arch_str, inputs_str=self.inputs_str, init_scale_str=self.init_scale_str, rngs=rng)
        # Based on print outputs, nnx.split() seems to return (GraphDef, State)
        # contrary to typical documentation (State, GraphDef).
        # So, assign accordingly:
        graphdef_returned, state_returned = nnx.split(dummy_cppn)

        # Now use state_returned for ParameterReshaper
        self.param_reshaper = ParameterReshaper(state_returned)
        self.n_params = self.param_reshaper.total_params

    def init(self, rng_key):
        # This method would initialize a new CPPN and return its flattened state.
        rngs = nnx.Rngs(params=rng_key)
        cppn_instance = self.cppn_module_class(arch=self.arch_str, inputs_str=self.inputs_str, init_scale_str=self.init_scale_str, rngs=rngs)
        # Based on print outputs, nnx.split() seems to return (GraphDef, State)
        _graphdef_returned, state_returned = nnx.split(cppn_instance)
        # print(f"FlattenCPPNParameters.init: state_returned before flatten_single: {state_returned}") # Removed
        return self.param_reshaper.flatten_single(state_returned)

    def generate_image(self, flat_state_params, img_size=256, return_features=False):
        # Reshape flat_state_params back into NNX module state
        # This is complex with NNX: you'd typically update an existing module's state
        # or create a new one and merge the state.

        # For now, this part is highly conceptual for NNX and would require
        # a more detailed implementation of how to apply a flat vector of parameters
        # to an NNX module's state.
        # One approach: create a new module and merge the reshaped state into it.
        rngs_for_reshaping = nnx.Rngs(params=jax.random.key(0)) # Dummy RNGs, actual params come from flat_state_params
        temp_cppn = self.cppn_module_class(arch=self.arch_str, inputs_str=self.inputs_str, init_scale_str=self.init_scale_str, rngs=rngs_for_reshaping)

        # The ParameterReshaper should give us the state in the correct structure
        reshaped_state = self.param_reshaper.reshape_single(flat_state_params)

        # Merge the reshaped state into the temporary CPPN instance
        nnx.update(temp_cppn, reshaped_state)

        return temp_cppn.generate_image(img_size=img_size, return_features=return_features)

class ParameterReshaper(object):
    def __init__(
        self,
        placeholder_params,
        n_devices: Optional[int] = None,
        verbose: bool = True,
    ):
        """Reshape flat parameters vectors into generation eval shape."""
        # Get network shape to reshape (for NNX, this is the module's state)
        self.placeholder_state = placeholder_params # Renamed for clarity with NNX

        # Set total parameters depending on type of placeholder state
        flat, self.unravel_pytree = flatten_util.ravel_pytree(
            self.placeholder_state
        )
        self.total_params = flat.shape[0]
        # Jitting unravel_pytree is fine as it operates on pytrees (state is a pytree)
        self.reshape_single = jax.jit(self.unravel_pytree)

        if n_devices is None:
            self.n_devices = jax.local_device_count()
        else:
            self.n_devices = n_devices
        if self.n_devices > 1 and verbose:
            print(
                f"ParameterReshaper: {self.n_devices} devices detected. Please"
                " make sure that the ES population size divides evenly across"
                " the number of devices to pmap/parallelize over."
            )

        if verbose:
            print(
                f"ParameterReshaper: {self.total_params} parameters detected"
                " for optimization."
            )

    def flatten_single(self, state_pytree): # Parameter 'x' renamed to 'state_pytree'
        """Reshaping pytree state (single) into flat array."""
        return ravel_pytree(state_pytree)



def ravel_pytree(pytree):
    leaves, _ = tree_flatten(pytree)
    flat, _ = vjp(ravel_list, *leaves)
    return flat


def ravel_list(*lst):
    return (
        jnp.concatenate([jnp.ravel(elt) for elt in lst])
        if lst
        else jnp.array([])
    )
