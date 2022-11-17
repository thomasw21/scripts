from typing import List
import torch
import torch._dynamo as dynamo
from transformers import AutoModel, AutoTokenizer

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("my_compiler() called with FX graph:")
    print(gm.graph)
    # gm.graph.print_tabular()
    return gm.forward  # return a python callable

@dynamo.optimize(my_compiler)
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    if b.sum() < 0:
        b = b * -1
    return x * b

def main():
    model = AutoModel.from_pretrained("bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    optimized_model = dynamo.optimize(my_compiler)(model)
    # print(type(optimized_model), isinstance(optimized_model, torch.fx.GraphModule))
    # model.forward = optimized_model.forward
    # optimized_model = dynamo.optimize("inductor")(model)

    for i in range(100):
        random_text = "potato"
        inputs_ids = tokenizer.encode(random_text, return_tensors="pt")

        optimized_model(inputs_ids)

    pass

if __name__ == "__main__":
    main()