import torch
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline

def test_pipeline_basic():
    """Test end-to-end vectorization pipeline"""
    pipeline = VectorizationPipeline(solver_type='torch')

    # Create simple test image
    image = torch.rand(1, 3, 128, 128)

    result = pipeline.vectorize(image)

    assert 'curves' in result
    assert 'reconstruction' in result
    assert result['reconstruction'].shape == (1, 3, 128, 128)
