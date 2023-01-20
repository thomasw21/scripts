import torch
from torch import nn

def main():
    # Create dummy model and optimizer
    model = nn.Linear(10, 20)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Somehow `state_dict` is empty here
    print(optimizer.state_dict())

    # Empty step to try to init
    optimizer.step()
    print(optimizer.state_dict())

    # Create dummy backward pass
    out = model(torch.randn(1, 10))
    out.mean().backward()
    optimizer.step()

    # Check state and store state_dict
    print(optimizer.state_dict())

if __name__ == "__main__":
    main()