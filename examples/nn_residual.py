import torch
import torch.nn as nn


class GELU(nn.Module):
    def forward(self, x):
        return (
            0.5
            * x
            * (
                1
                + torch.tanh(
                    torch.sqrt(torch.tensor(2.0 / torch.pi))
                    * (x + 0.044715 * torch.pow(x, 3))
                )
            )
        )

"""
TLDR here -
 Gradients of deep NNs tend to "vanish" as they propogate back to earlier layers (TODO -- would be good to establish a
 mathematical intuition for _why_)

But observably true in this example regardless; residual seems to fix.

"""

class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            layer_out = layer(x)
            if self.use_shortcut and x.shape == layer_out.shape:
                x = x + layer_out
            else:
                x = layer_out

        return x


def print_grads(model, x):
    output = model(x)
    target = torch.tensor([[0.]])
    loss  = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")


if __name__ == "__main__":
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([1., 0., -1.])
    torch.manual_seed(123)
    model_wo_short = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )

    print_grads(model_wo_short, sample_input)

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=True
    )
    print_grads(model_with_shortcut, sample_input)

    # NOTE -- continue with 4.6 Transformer block
    # Question to ask; why LayerNorm _and_ residual both needed for exploding/vanishing