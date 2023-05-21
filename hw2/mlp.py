from itertools import chain
from typing import List, Tuple

import torch
from torch import Tensor, nn
from typing import Union, Sequence
from collections import defaultdict

ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softmax": nn.Softmax,
    "logsoftmax": nn.LogSoftmax,
    "lrelu": nn.LeakyReLU,
    "none": nn.Identity,
    None: nn.Identity,
}

# Default keyword arguments to pass to activation class constructors, e.g.
# activation_cls(**ACTIVATION_DEFAULT_KWARGS[name])
ACTIVATION_DEFAULT_KWARGS = defaultdict(
    dict,
    {
        ###
        "softmax": dict(dim=1),
        "logsoftmax": dict(dim=1),
    },
)


class MLP(nn.Module):
    """
    A general-purpose MLP.
    """

    def __init__(
            self, in_dim: int, dims: Sequence[int], nonlins: Sequence[Union[str, nn.Module]]
    ):
        """
        :param in_dim: Input dimension.
        :param dims: Hidden dimensions, including output dimension.
        :param nonlins: Non-linearities to apply after each one of the hidden
            dimensions.
            Can be either a sequence of strings which are keys in the ACTIVATIONS
            dict, or instances of nn.Module (e.g. an instance of nn.ReLU()).
            Length should match 'dims'.
        """
        assert len(nonlins) == len(dims)
        self.in_dim = in_dim
        self.out_dim = dims[-1]

        # TODO:
        #  - Initialize the layers according to the requested dimensions. Use
        #    either nn.Linear layers or create W, b tensors per layer and wrap them
        #    with nn.Parameter.
        #  - Either instantiate the activations based on their name or use the provided
        #    instances.
        # ====== YOUR CODE: ======
        super().__init__()
        self.layers: List[nn.Linear] = []
        self.activations: List[nn.Module] = []
        self._create_layers([in_dim] + dims)
        self._create_activations(nonlins)

        layers_and_activations_list = list(chain.from_iterable(zip(self.layers, self.activations)))
        self.fc_layers = nn.Sequential(*layers_and_activations_list)

        # ========================

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: An input tensor, of shape (N, D) containing N samples with D features.
        :return: An output tensor of shape (N, D_out) where D_out is the output dim.
        """
        # TODO: Implement the model's forward pass. Make sure the input and output
        #  shapes are as expected.
        # ====== YOUR CODE: ======
        # assert x.shape[-1] == self.in_dim, f"Input shape was {x.shape[-1]} but expected {self.in_dim}"

        # i = 0
        # while i < len(self.fc_layers):
        #     layer = self.fc_layers[i]
        #     activation = self.fc_layers[i + 1]
        #     x = activation(layer(x))
        #     i += 2

        # assert x.shape[-1] == self.out_dim, f"Output shape was {x.shape[-1]} but expected {self.out_dim}"
        return self.fc_layers(x)
        # return x
        # ========================

    def _create_layers(self, dims) -> None:
        hidden_layers_dims_except_output = dims[:-1]
        hidden_layers_dims_except_input = dims[1:]
        for hidden_in_dim, hidden_out_dim in zip(hidden_layers_dims_except_output, hidden_layers_dims_except_input):
            layer = nn.Linear(hidden_in_dim, hidden_out_dim, bias=True)
            # nn.init.normal_(layer.weight, 0, 1)
            self.layers.append(layer)

    def _create_activations(self, nonLins) -> None:
        activation = None
        for nonLinearity in nonLins:
            if isinstance(nonLinearity, str):
                activation_cls = ACTIVATIONS[nonLinearity]
                activation_kwargs = ACTIVATION_DEFAULT_KWARGS[nonLinearity]
                activation = activation_cls(**activation_kwargs)
            elif isinstance(nonLinearity, nn.Module):
                activation = nonLinearity

            self.activations.append(activation)
