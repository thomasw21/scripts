import torch
from torch import nn


def main():
    # Create dummy model and optimizer
    model1 = nn.Linear(2, 5)
    model2 = nn.Linear(3, 5)

    optimizer = torch.optim.Adam(
        [
            {"lr": 1e-3, "params": model1.parameters()},
            {"lr": 1e-4, "params": model2.parameters()}
        ],
        lr=1e-3)

    # Somehow `state_dict` is empty here
    print(optimizer.state_dict())

    # Empty step to try to init
    optimizer.step()
    print(optimizer.state_dict())

    # Create dummy backward pass
    out = model1(torch.randn(1, 2)) + model2(torch.randn(1, 3))
    out.mean().backward()
    optimizer.step()

    # Check state and store state_dict
    print(optimizer.state_dict())

    # Check param groups
    print(optimizer.param_groups)
    pass

if __name__ == "__main__":
    main()