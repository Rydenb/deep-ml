import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    # Your implementation here
    m = torch.nn.Sigmoid()
    input = torch.tensor(z)
    output = m(input)
    return output.item()
