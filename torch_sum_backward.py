import torch


class Print(torch.autograd.Function):
    @staticmethod
    def forward(ctx, id: str, x):
        ctx.id = id
        return x

    @staticmethod
    def backward(ctx, grad):
        print(ctx.id)
        return None, grad

def main():
    l = []
    for i in range(10):
        x = torch.randn(1, requires_grad=True)
        x = Print.apply(str(i), x)
        l.append(x)
    l.reverse()
    sum(l).backward()


if __name__ == "__main__":
    main()

