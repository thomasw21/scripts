import math

import torch


def main():
    hidden_dim = 768
    num_heads = 16
    assert hidden_dim % num_heads == 0
    head_dim = hidden_dim // num_heads

    epochs = 10000
    learning_rate = 1e-1

    ORIGINAL_W_Qs = torch.stack([torch.rand(head_dim, hidden_dim) for _ in range(num_heads)], dim=0)
    ORIGINAL_W_Ks = torch.stack([torch.rand(head_dim, hidden_dim) for _ in range(num_heads)], dim=0)
    ORIGINAL_W_Vs = torch.stack([torch.rand(head_dim, hidden_dim) for _ in range(num_heads)], dim=0)

    """
    Question 1: Can I find  and W_V in R^{hidden_dim x num_heads} 
     - W_K in R^{hidden_dim x num_heads}
     - (W_Q)_i in (R^{hidden_dim x num_heads}) ^ {num_heads}
    s.t
     - for all i, (ORIGINAL_W_Q)_i ^ T * (ORIGINAL_W_K)_i = (W_Q)_i ^ T * W_K

    Answer: Maybe?

    Question 2: Can we do the same for value:
     - W_V in R^{hidden_dim x num_heads}
    s.t
     - for all i, (ORIGINAL_W_V)_i = W_V

    Answer: No, the best we can do is to take the average and it's closest one can get using L2 norm
    """

    W_Qs = torch.stack([torch.rand(head_dim, hidden_dim) for _ in range(num_heads)], dim=0)
    W_K = torch.rand(head_dim, hidden_dim)

    # Make them trainable
    W_Qs.requires_grad = True
    W_K.requires_grad = True

    optimizer = torch.optim.Adam([W_Qs, W_K], lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 1 / math.sqrt(iter + 1))

    # Precomputed ORIGINAL_W_Qs 8 ORIGINAL_W_Ks
    precomputed_orignal_QK = torch.bmm(ORIGINAL_W_Qs.permute(0, 2, 1), ORIGINAL_W_Ks)

    for it in range(epochs):
        current_approximation = torch.bmm(W_Qs.permute(0, 2, 1), W_K[None, ...].expand(num_heads, -1, -1))

        loss = torch.sum(
            (
                    current_approximation - precomputed_orignal_QK
            ) ** 2 / (hidden_dim ** 2)
        )

        if it % 100 == 0:
            print(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


if __name__ == "__main__":
    main()