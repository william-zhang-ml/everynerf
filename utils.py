"""Utility code not strictly related to neural radiance fields. """
from typing import Union
import PIL
import requests
import torch
from torchvision import transforms


def url_to_image(
    url: str,
    to_tensor: bool = False,
    timeout: int = 10
) -> Union[PIL.Image.Image, torch.Tensor]:
    """Get an image from online.

    Args:
        url (str): path to image
        to_tensor (bool, optional): cast-to-tensor flag. Defaults to False.
        timeout (int, optional): max seconds to wait. Defaults to 10.

    Returns:
        Union[PIL.Image.Image, torch.Tensor]: image
    """
    stream = requests.get(url, timeout=timeout, stream=True).raw
    img = PIL.Image.open(stream)
    if to_tensor:
        img = transforms.ToTensor()(img)
    return img
