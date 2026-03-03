# Implementation Notes

## Architecture Overview

This implementation follows the paper's algorithm with some simplifications for balance between accuracy and speed:

### Simplifications Made

1. **Curve Extraction**: Uses simple gradient-based edge detection instead of full Canny with hysteresis
2. **Solver**: Initial implementation uses direct color sampling; can be enhanced with full sparse linear system
3. **Diffusion**: Uses iterative Laplacian diffusion instead of solving Poisson equation directly

### Solver Comparison

Two solver backends are provided:

**PyTorch Solver**
- Pros: GPU acceleration, integrates with PyTorch pipeline
- Cons: Sparse solver support less mature than SciPy

**SciPy Solver**
- Pros: Mature sparse solvers, well-tested
- Cons: CPU-only, requires data transfer from GPU

### Future Enhancements

1. Implement full Laplacian/Bilaplacian weight solving (Section 5.2)
2. Add hierarchical curve consistency (Section 4.3)
3. Implement Douglas-Peucker curve simplification
4. Add bilateral filtering for structure-preserving prefiltering
5. Optimize curve rasterization with GPU kernels

## Testing

Run full test suite:
```bash
pytest tests/ -v
```

Compare solvers:
```bash
pytest tests/test_solver_comparison.py -v -s
```

## Usage for Comparison Experiments

```python
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline
import torch

# Test both solvers
for solver in ['torch', 'scipy']:
    pipeline = VectorizationPipeline(solver_type=solver)
    result = pipeline.vectorize(your_image)
    # Compare results...
```
