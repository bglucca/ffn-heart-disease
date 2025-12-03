import torch.nn as nn


class ClassifierNetwork(nn.Module):  # pylint: disable=too-few-public-methods
    """
    A configurable feed-forward neural network (multi-layer perceptron) classifier.

    Args:
        n_hidden_layers (int): Number of fully connected hidden layers.
        input_size (int): Dimensionality of the input features.
        hidden_size (int): Number of neurons in each hidden layer.
        output_size (int): Number of output classes.
        activation_fn (nn.Module, optional): Activation function for hidden layers (default: nn.ReLU()).
        output_activation_fn (nn.Module, optional): Activation function for the output layer (default: nn.Softmax(dim=1)).

    The architecture consists of:
        - An input layer mapping input_size to hidden_size.
        - n_hidden_layers fully connected layers of hidden_size neurons.
        - An output layer mapping hidden_size to output_size.
        - Applies activation_fn after each hidden layer and output_activation_fn to output.
    """
    def __init__(
        self,
        n_hidden_layers: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation_fn: nn.Module = nn.ReLU()
    ):
        super().__init__()

        # Layers
        self.layers = nn.ModuleDict()
        self.layers['input'] = nn.Linear(input_size, hidden_size)
        for i in range(n_hidden_layers):
            self.layers[f'hidden_{i}'] = nn.Linear(hidden_size, hidden_size)

        self.layers['output'] = nn.Linear(hidden_size, output_size)

        # Activation functions
        self.activation_fn = activation_fn

    def forward(self, x):
        x = self.activation_fn(self.layers['input'](x))
        for key, layer in self.layers.items():
            if key.startswith('hidden_'):
                x = self.activation_fn(layer(x))
        return self.layers['output'](x)
