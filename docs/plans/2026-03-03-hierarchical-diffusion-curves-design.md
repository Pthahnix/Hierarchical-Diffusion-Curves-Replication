# Hierarchical Diffusion Curves Implementation Design

## Overview

Implementation of "Hierarchical Diffusion Curves for Accurate Automatic Image Vectorization" paper for comparison experiments. The system will convert raster images to vector representations using multi-scale diffusion curves.

## Requirements

- Full image vectorization pipeline
- Hierarchical curve extraction at multiple scales
- Dual solver implementation (PyTorch and SciPy) for performance comparison
- Curve rendering and reconstruction
- Python implementation using PyTorch as primary framework
- Single image processing interface
- Balance between accuracy and speed

## Architecture

### Module Structure

```
hierarchical_diffusion_curves/
├── prefilter.py           # Structure-preserving image prefiltering
├── curve_extraction.py    # Laplacian domain curve extraction
├── hierarchy.py           # Hierarchical curve consistency
├── solvers/
│   ├── base.py           # Abstract solver interface
│   ├── torch_solver.py   # PyTorch sparse solver
│   └── scipy_solver.py   # SciPy sparse solver
├── renderer.py           # Diffusion curve rendering
├── pipeline.py           # End-to-end pipeline
└── utils.py              # Helper functions
```

### Data Structures

```python
@dataclass
class DiffusionCurve:
    points: torch.Tensor      # (N, 2) curve control points
    colors_left: torch.Tensor # (N, 3) left-side colors
    colors_right: torch.Tensor # (N, 3) right-side colors
    blur_left: torch.Tensor   # (N,) left blur radius
    blur_right: torch.Tensor  # (N,) right blur radius
    scale_level: int          # pyramid level

@dataclass
class HierarchicalCurves:
    curves: List[List[DiffusionCurve]]  # [level][curve_idx]
    image_pyramid: List[torch.Tensor]   # Gaussian pyramid
    laplacian_pyramid: List[torch.Tensor] # Laplacian pyramid
```

### Solver Interface

```python
class CurveSolver(ABC):
    @abstractmethod
    def solve_weights(self, curves, target_laplacian):
        """Solve curve weights to approximate target Laplacian"""
        pass
```

## Algorithm Pipeline

### 1. Prefiltering (Section 4.1)

- Build Gaussian pyramid (3-5 levels)
- Apply structure-preserving filter at each level (bilateral or guided filter)
- Compute Laplacian pyramid for curve extraction

### 2. Curve Extraction (Section 4.2)

For each pyramid level:
- Compute gradient magnitude of Laplacian image
- Non-maximum suppression for edge detection
- Canny edge detection + curve tracing
- Sample curve control points with adaptive spacing
- Extract curves from coarse to fine

### 3. Hierarchical Consistency (Section 4.3)

- Upsample coarse-level curves to fine levels
- Search for corresponding curve segments in fine levels
- Establish cross-level curve correspondence
- Merge redundant curves

### 4. Weight Solving (Section 5)

- Build sparse linear system Ax = b
  - A: Laplacian/Bilaplacian diffusion operator
  - b: target Laplacian pyramid values
  - x: curve color weights
- Solve using torch or scipy solver

### 5. Rendering and Reconstruction

- Reconstruct image from curve parameters
- Apply Laplacian diffusion
- Multi-scale fusion

## Implementation Details

### Prefiltering Module

- Use kornia for Gaussian pyramid (pure PyTorch)
- Bilateral filter: kornia.filters.bilateral_blur
- Laplacian pyramid: current_level - upsample(next_level)

### Curve Extraction Module

- Gradient: torch.gradient or Sobel operator
- Edge detection: kornia.filters.canny or custom implementation
- Curve tracing: greedy connection algorithm for continuity
- Control point sampling: Douglas-Peucker algorithm for simplification

### Solver Module

**PyTorch version:**
- torch.sparse.mm + torch.linalg.solve
- GPU acceleration

**SciPy version:**
- scipy.sparse.linalg.spsolve or iterative solvers (cg/gmres)
- CPU-based, mature and stable

Unified interface for easy performance comparison.

### Rendering Module

- Curve rasterization: Bresenham algorithm or differentiable rasterization
- Laplacian diffusion: solve Poisson equation with sparse Laplacian matrix
- Bilaplacian diffusion: 4th-order diffusion for smoother transitions

## Dependencies

- torch, torchvision
- kornia (image processing)
- scipy (optional solver)
- numpy, pillow (utilities)

## Error Handling

- Input validation: check image format, size, value range
- Curve extraction failure: interpolate from adjacent levels if no curves found
- Solver failure: fallback to least squares if sparse solve doesn't converge
- Memory management: automatic chunking for large images

## Testing Strategy

- Unit tests: test each module independently
- Integration tests: end-to-end pipeline validation
- Comparison tests: simple test images (gradients, geometric shapes) for accuracy
- Performance tests: benchmark time and memory for both solvers

## Output Format

- Curve data: JSON format (for visualization and debugging)
- Reconstructed image: PNG format
- Intermediate results: optional save of pyramids, edge maps for debugging

## Implementation Approach

Modular design with two solver backends allows flexible testing:
1. Implement core pipeline with PyTorch
2. Create abstract solver interface
3. Implement both torch and scipy solvers
4. Provide unified API for comparison experiments
5. User can benchmark and choose the better solver for their use case
