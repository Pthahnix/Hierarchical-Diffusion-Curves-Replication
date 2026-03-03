import torch
from hierarchical_diffusion_curves.curve_extraction import detect_edges

def test_edge_detection():
    """Test edge detection on Laplacian image"""
    laplacian = torch.rand(1, 3, 128, 128)
    edges = detect_edges(laplacian, threshold=0.1)

    assert edges.shape == (1, 1, 128, 128)
    assert edges.dtype == torch.float32
    assert edges.min() >= 0.0 and edges.max() <= 1.0
