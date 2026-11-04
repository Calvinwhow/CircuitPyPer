"""Optional dependency access for the neural optimization path."""


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("Neural optimization requires PyTorch. Install calvin_utils[neural] or `pip install torch`.") from error
    return torch
