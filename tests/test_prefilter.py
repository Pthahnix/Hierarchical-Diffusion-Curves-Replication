import torch
from hierarchical_diffusion_curves.prefilter import build_gaussian_pyramid

def test_gaussian_pyramid():
    """Test building Gaussian pyramid"""
    image = torch.rand(1, 3, 256, 256)
    pyramid = build_gaussian_pyramid(image, num_levels=3)

    assert len(pyramid) == 3
    assert pyramid[0].shape == (1, 3, 256, 256)
    assert pyramid[1].shape == (1, 3, 128, 128)
    assert pyramid[2].shape == (1, 3, 64, 64)
