import torch
from hierarchical_diffusion_curves.curve_extraction import detect_edges, trace_curves

def test_edge_detection():
    """Test edge detection on Laplacian image"""
    laplacian = torch.rand(1, 3, 128, 128)
    edges = detect_edges(laplacian, threshold=0.1)

    assert edges.shape == (1, 1, 128, 128)
    assert edges.dtype == torch.float32
    assert edges.min() >= 0.0 and edges.max() <= 1.0

def test_curve_tracing():
    """Test tracing curves from edge map"""
    # Create simple edge map with a line
    edges = torch.zeros(1, 1, 64, 64)
    edges[0, 0, 32, 10:50] = 1.0  # Horizontal line

    curves = trace_curves(edges, min_length=5)

    assert len(curves) > 0
    assert all(len(curve) >= 5 for curve in curves)
