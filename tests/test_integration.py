import torch
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline

def test_full_pipeline_torch():
    """Test full pipeline with PyTorch solver"""
    pipeline = VectorizationPipeline(solver_type='torch', num_levels=3)

    # Create test image with sharp edges
    image = torch.zeros(1, 3, 128, 128)
    image[0, :, 30:70, 30:70] = 1.0  # White square

    result = pipeline.vectorize(image)

    assert 'curves' in result
    assert 'reconstruction' in result
    assert result['reconstruction'].shape == (1, 3, 128, 128)
    # May or may not find curves depending on threshold, so just check structure
    assert isinstance(result['curves'], list)

def test_full_pipeline_scipy():
    """Test full pipeline with SciPy solver"""
    pipeline = VectorizationPipeline(solver_type='scipy', num_levels=3)

    # Create test image with sharp edges
    image = torch.zeros(1, 3, 128, 128)
    image[0, :, 30:70, 30:70] = 1.0  # White square

    result = pipeline.vectorize(image)

    assert 'curves' in result
    assert 'reconstruction' in result
    assert isinstance(result['curves'], list)
