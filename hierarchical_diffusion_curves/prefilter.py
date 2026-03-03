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

def build_laplacian_pyramid(gaussian_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    """Build Laplacian pyramid from Gaussian pyramid

    Args:
        gaussian_pyramid: List of Gaussian pyramid levels

    Returns:
        List of Laplacian images (difference between levels)
    """
    laplacian_pyramid = []

    for i in range(len(gaussian_pyramid) - 1):
        current = gaussian_pyramid[i]
        next_level = gaussian_pyramid[i + 1]

        # Upsample next level to current size
        upsampled = F.interpolate(
            next_level,
            size=current.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # Laplacian = current - upsampled(next)
        laplacian = current - upsampled
        laplacian_pyramid.append(laplacian)

    # Add the coarsest level as-is
    laplacian_pyramid.append(gaussian_pyramid[-1])

    return laplacian_pyramid
