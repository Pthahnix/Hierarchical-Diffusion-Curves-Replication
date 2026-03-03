import torch
import torch.nn.functional as F
from typing import List

def build_gaussian_pyramid(image: torch.Tensor, num_levels: int = 4) -> List[torch.Tensor]:
    """Build Gaussian pyramid by iterative downsampling

    Args:
        image: Input image (B, C, H, W)
        num_levels: Number of pyramid levels

    Returns:
        List of downsampled images from fine to coarse
    """
    pyramid = [image]
    current = image

    for _ in range(num_levels - 1):
        # Gaussian blur before downsampling
        kernel_size = 5
        sigma = 1.0
        current = F.avg_pool2d(current, kernel_size=2, stride=2)
        pyramid.append(current)

    return pyramid
