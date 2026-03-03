# Hierarchical Diffusion Curves Repliction

Implementation of "Hierarchical Diffusion Curves for Accurate Automatic Image Vectorization" for comparison experiments.

## Installation

```bash
pip install -e .
```

## Usage

```python
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline
import torch

# Create pipeline with PyTorch solver
pipeline = VectorizationPipeline(solver_type='torch', num_levels=3)

# Vectorize image
image = torch.rand(1, 3, 256, 256)
result = pipeline.vectorize(image)

# Access curves and reconstruction
curves = result['curves']
reconstruction = result['reconstruction']
```

## Command Line

```bash
python examples/vectorize_image.py input.jpg output.jpg --solver torch
```

## Solver Comparison

Two solver backends are available:
- `torch`: PyTorch sparse solver (GPU-accelerated)
- `scipy`: SciPy sparse solver (CPU-based, mature)

Compare performance:
```python
import time

for solver_type in ['torch', 'scipy']:
    pipeline = VectorizationPipeline(solver_type=solver_type)
    start = time.time()
    result = pipeline.vectorize(image)
    print(f"{solver_type}: {time.time() - start:.2f}s")
```
