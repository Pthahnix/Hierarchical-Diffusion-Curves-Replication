import torch
from hierarchical_diffusion_curves.renderer import rasterize_curve

def test_rasterize_curve():
    """Test rasterizing a curve to image"""
    curve_points = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 10.0]])
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    image = rasterize_curve(curve_points, colors, image_size=(64, 64))

    assert image.shape == (3, 64, 64)
    assert image.max() <= 1.0
    assert image.min() >= 0.0
