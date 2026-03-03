import torch
import time
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline

def test_solver_comparison():
    """Compare PyTorch and SciPy solvers"""
    image = torch.rand(1, 3, 128, 128)

    results = {}
    for solver_type in ['torch', 'scipy']:
        pipeline = VectorizationPipeline(solver_type=solver_type, num_levels=3)

        start = time.time()
        result = pipeline.vectorize(image)
        elapsed = time.time() - start

        results[solver_type] = {
            'time': elapsed,
            'num_curves': len(result['curves']),
            'reconstruction': result['reconstruction']
        }

        print(f"{solver_type} solver: {elapsed:.3f}s, {len(result['curves'])} curves")

    # Both should produce similar number of curves
    assert abs(results['torch']['num_curves'] - results['scipy']['num_curves']) < 10
