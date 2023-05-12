import torch
from torch import nn

def main():
    # First version of weight sharing
    model1 = nn.Linear(2, 3)
    model2 = nn.Linear(2,3)
    model2.weight.data = model1.weight.data
    model2.bias.data = model1.bias.data
    optimizer = torch.optim.Adam([*model1.parameters(), *model2.parameters()], lr=1e-3)

    print(optimizer.state_dict())
    print(optimizer.param_groups)

    # Try get parameter indexing
    state_dict = optimizer.state_dict()
    assert len(optimizer.param_groups) == len(state_dict["param_groups"])
    id_to_index = {}
    for real_param_group, index_param_group in zip(optimizer.param_groups, state_dict["param_groups"]):
        indices = index_param_group["params"]
        parameters = real_param_group["params"]
        assert len(indices) == len(parameters)
        for params, id_ in zip(parameters, indices):
            id_to_index[id(params)] = id_
    print(id_to_index)

    # Second version of weight sharing
    model1_bis = nn.Linear(2, 3)
    model2_bis = nn.Linear(2, 3)
    model2_bis.weight.data = model1.weight.data
    model2_bis.bias.data = model1.bias.data
    optimizer_bis = torch.optim.Adam([*model1.parameters(), *model2.parameters()], lr=1e-3)


    input_random_1 = torch.randn(1,2)
    input_random_2 = torch.randn(1,2)

    # Create dummy backward pass
    out = model1(torch.randn(1, 2)) + model2(torch.randn(1, 2))
    out.mean().backward()

    assert model1.weight.grad != model2.weight.grad

    # # Sync gradients
    # model_weight_grad = model1.weight.grad + model2.weight.grad
    # model_bias_grad = model1.bias.grad + model2.bias.grad
    # model1.weight.grad = model_weight_grad
    # model2.weight.grad = model_weight_grad
    # model1.bias.grad = model_bias_grad
    # model2.bias.grad = model_bias_grad

    optimizer.step()

    # Check state and store state_dict
    print(optimizer.state_dict())
    torch.testing.assert_close(model1.weight, model2.weight)
    torch.testing.assert_close(model1.bias, model2.bias)

if __name__ == "__main__":
    main()