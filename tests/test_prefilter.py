import torch
from hierarchical_diffusion_curves.prefilter import build_gaussian_pyramid, build_laplacian_pyramid

def test_gaussian_pyramid():
    """Test building Gaussian pyramid"""
    image = torch.rand(1, 3, 256, 256)
    pyramid = build_gaussian_pyramid(image, num_levels=3)

    assert len(pyramid) == 3
    assert pyramid[0].shape == (1, 3, 256, 256)
    assert pyramid[1].shape == (1, 3, 128, 128)
    assert pyramid[2].shape == (1, 3, 64, 64)

def test_laplacian_pyramid():
    """Test building Laplacian pyramid"""
    image = torch.rand(1, 3, 256, 256)
    gaussian_pyr = build_gaussian_pyramid(image, num_levels=3)
    laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)

    assert len(laplacian_pyr) == 3
    # Laplacian = current - upsample(next)
    assert laplacian_pyr[0].shape == (1, 3, 256, 256)
    assert laplacian_pyr[1].shape == (1, 3, 128, 128)
