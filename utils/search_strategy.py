import os

import torch


def search_strategy() -> str:
    """
    Check if system has CPU, GPU or MPS.
    Returns: 'cpu', 'cuda', torch.device('mps')
    """

    # Detect hardware
    device = 'cpu'
    try:
        device = torch.device('mps')
    except AttributeError:
        device = 'cuda' if torch.cuda.is_available() else device
    return device, os.cpu_count()


if __name__ == '__main__':
    print(search_strategy())
