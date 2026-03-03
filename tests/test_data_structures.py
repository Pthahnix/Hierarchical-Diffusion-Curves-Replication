import torch
from hierarchical_diffusion_curves.data_structures import DiffusionCurve

def test_diffusion_curve_creation():
    """Test creating a diffusion curve"""
    points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    colors_left = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    colors_right = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
    blur_left = torch.tensor([1.0, 1.5])
    blur_right = torch.tensor([1.0, 1.5])

    curve = DiffusionCurve(
        points=points,
        colors_left=colors_left,
        colors_right=colors_right,
        blur_left=blur_left,
        blur_right=blur_right,
        scale_level=0
    )

    assert curve.points.shape == (2, 2)
    assert curve.scale_level == 0
