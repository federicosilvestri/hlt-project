import torch


def search_strategy() -> str:
    """
    Check if system has CPU, GPU or GPU.
    Returns: 'cpu', 'cuda'
    """

    # Detect hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


if __name__ == "__main__":
    print(search_strategy())
