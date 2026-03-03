import torch
from hierarchical_diffusion_curves.renderer import rasterize_curve, apply_diffusion

def test_rasterize_curve():
    """Test rasterizing a curve to image"""
    curve_points = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 10.0]])
    colors = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    image = rasterize_curve(curve_points, colors, image_size=(64, 64))

    assert image.shape == (3, 64, 64)
    assert image.max() <= 1.0
    assert image.min() >= 0.0

def test_apply_diffusion():
    """Test applying Laplacian diffusion"""
    # Create image with sparse values
    image = torch.zeros(3, 64, 64)
    image[:, 32, 32] = 1.0

    diffused = apply_diffusion(image, num_iterations=10)

    assert diffused.shape == (3, 64, 64)
    # Diffusion should spread values
    assert diffused[:, 30, 30].sum() > 0
