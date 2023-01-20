import torch


def main(device: torch.device = torch.device("cpu")):
    x = torch.randn(2,3,5,7, device=device)
    y = x.to(memory_format=torch.channels_last)

    # Assert that they don't share memory
    do_share_memory = x.data_ptr() == y.data_ptr()
    print(f"Channels Last shares memory: {do_share_memory}")

    # Assert that channels last are not contiguous
    print(f"Channels Last is contiguous: {y.is_contiguous()}")

    # Data storage is equivalent to contiguous NHWC
    z = x.permute(0,2,3,1).contiguous()
    is_storage_equal = all(z_data == y_data for z_data, y_data in zip(z.storage(), y.storage()))
    print(f"Channels Last storage is the same as the NHWC format: {is_storage_equal}")

    """
    This begs the question: why does pytorch not expose NHWC convolutions then? IMO it's a weird API
    """

    # Assume that x is stored in NHWC format
    N, H, W, C = x.shape
    conv1 = torch.nn.Conv2d(C, C, 1).to(device)
    a = conv1(x.permute(0, 3, 1, 2))

    assert (N, C, H , W) == a.shape

    # Assert that `z` is channels_last contiguous
    print(f"Input of convolution is channels_last contiguous: {x.is_contiguous(memory_format=torch.channels_last)}")
    print(f"Output of convolution is channels_last contiguous by default: {a.is_contiguous(memory_format=torch.channels_last)}")

    # Assume that x is stored in NCHW format
    N, C, H, W = x.shape
    conv1 = torch.nn.Conv2d(C, C, 1).to(device)
    a = conv1(x)

    assert (N, C, H, W) == a.shape

    # Assert that `z` is channels_last contiguous
    print(f"Input of convolution is channels_last contiguous: {x.is_contiguous(memory_format=torch.channels_last)}")
    print(
        f"Output of convolution is channels_last contiguous by default: {a.is_contiguous(memory_format=torch.channels_last)}")

    # Workaround
    N, H, W, C = x.shape
    conv1 = torch.nn.Conv2d(C, C, 1).to(device)
    a = conv1(x.permute(0, 3, 1, 2)).permute(0, 2,3,1)

    assert (N, H, W, C) == a.shape

    # Assert that `z` is channels_last contiguous
    print(f"Input of convolution is channels_last contiguous: {x.is_contiguous(memory_format=torch.contiguous_format)}")
    print(
        f"Output of convolution is channels_last contiguous by default: {a.is_contiguous(memory_format=torch.contiguous_format)}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    main(device=device)