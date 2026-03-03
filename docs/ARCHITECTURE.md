# Architecture Guide

## Overview

This document explains the architectural design of the hierarchical diffusion curves implementation, including design decisions, module interactions, and extensibility points.

## System Architecture

### High-Level Flow

```
Input Image (B, C, H, W)
    ↓
[Prefiltering]
    ↓
Gaussian Pyramid → Laplacian Pyramid
    ↓
[Curve Extraction] (per level)
    ↓
Edge Detection → Curve Tracing
    ↓
[Weight Solving]
    ↓
Solver (PyTorch/SciPy) → Curve Weights
    ↓
[Rendering]
    ↓
Rasterization → Diffusion → Reconstruction
```

### Module Dependency Graph

```
pipeline.py
    ├── prefilter.py
    │   └── torch.nn.functional
    ├── curve_extraction.py
    │   ├── torch
    │   └── numpy
    ├── solvers/
    │   ├── base.py (abstract)
    │   ├── torch_solver.py
    │   │   └── torch
    │   └── scipy_solver.py
    │       ├── scipy.sparse
    │       └── numpy
    ├── renderer.py
    │   └── torch.nn.functional
    └── data_structures.py
        └── torch
```

## Module Design

### 1. Prefiltering Module (`prefilter.py`)

**Purpose:** Multi-scale image decomposition for hierarchical processing.

**Design Rationale:**
- Gaussian pyramid provides coarse-to-fine representation
- Laplacian pyramid captures detail at each scale
- Enables processing different frequency bands separately

**Key Functions:**

```python
def build_gaussian_pyramid(image: torch.Tensor, num_levels: int = 4) -> List[torch.Tensor]:
    """
    Iteratively downsample image with blur.

    Design choice: Use avg_pool2d for simplicity.
    Alternative: Could use kornia.filters.gaussian_blur2d for better quality.
    """
```

```python
def build_laplacian_pyramid(gaussian_pyramid: List[torch.Tensor]) -> List[torch.Tensor]:
    """
    Compute difference between levels.

    Design choice: Bilinear interpolation for upsampling.
    Alternative: Could use learned upsampling for better reconstruction.
    """
```

**Trade-offs:**
- ✅ Simple and fast
- ✅ Works well for most images
- ❌ May lose fine details with aggressive downsampling
- ❌ No edge-aware filtering

### 2. Curve Extraction Module (`curve_extraction.py`)

**Purpose:** Detect edges and trace them into continuous curves.

**Design Rationale:**
- Edges in Laplacian images indicate color discontinuities
- Connected components form natural curve primitives
- Minimum length filtering removes noise

**Key Functions:**

```python
def detect_edges(laplacian: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """
    Sobel gradient-based edge detection.

    Design choice: Simple gradient magnitude thresholding.
    Why: Fast, differentiable, good enough for most cases.

    Alternative approaches considered:
    - Canny edge detection (more accurate but not differentiable)
    - Learned edge detection (requires training data)
    """
```

```python
def trace_curves(edges: torch.Tensor, min_length: int = 10) -> List[torch.Tensor]:
    """
    8-connected component tracing.

    Design choice: Depth-first search with visited tracking.
    Why: Simple, guarantees connected curves.

    Limitations:
    - No curve simplification (could add Douglas-Peucker)
    - No curve smoothing (could add B-spline fitting)
    - No junction handling (curves may branch)
    """
```

**Trade-offs:**
- ✅ Fast and simple
- ✅ Produces connected curves
- ❌ No curve simplification (many points)
- ❌ No smoothness guarantees
- ❌ Sensitive to threshold parameter

### 3. Solver Module (`solvers/`)

**Purpose:** Compute color weights for curves to approximate target Laplacian.

**Design Rationale:**
- Abstract interface allows easy comparison between implementations
- Pluggable architecture supports future solver additions
- Both solvers implement same interface for fair benchmarking

#### Abstract Base Class (`base.py`)

```python
class CurveSolver(ABC):
    """
    Design pattern: Strategy pattern for solver selection.

    Why abstract base class:
    - Enforces consistent interface
    - Enables polymorphism
    - Documents expected behavior
    """

    @abstractmethod
    def solve_weights(self, curves, target_laplacian, image_size):
        """
        Contract: Given curves and target, return color weights.

        Implementations must:
        1. Handle empty curve lists gracefully
        2. Return weights matching curve point counts
        3. Produce RGB colors in [0, 1] range (or unbounded for HDR)
        """
```

#### PyTorch Solver (`torch_solver.py`)

```python
class TorchSolver(CurveSolver):
    """
    Design choice: Direct color sampling from target.

    Current implementation:
    - Sample target colors at curve point locations
    - Simple and fast
    - No optimization required

    Future enhancement:
    - Solve sparse linear system: Ax = b
    - A: curve influence matrix (sparse)
    - b: target Laplacian values
    - x: curve color weights
    - Use torch.sparse for GPU acceleration
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Design choice: Auto-detect GPU availability.
        Why: Maximize performance without user configuration.
        """
```

**Trade-offs:**
- ✅ GPU acceleration potential
- ✅ Integrates seamlessly with PyTorch pipeline
- ✅ Differentiable (could enable end-to-end learning)
- ❌ Sparse solver support less mature than SciPy
- ❌ GPU overhead for small problems

#### SciPy Solver (`scipy_solver.py`)

```python
class ScipySolver(CurveSolver):
    """
    Design choice: Use SciPy's mature sparse solvers.

    Advantages:
    - Well-tested sparse linear algebra
    - Multiple solver algorithms available
    - Excellent documentation

    Disadvantages:
    - CPU-only (no GPU)
    - Requires data transfer from GPU tensors
    """
```

**Trade-offs:**
- ✅ Mature, well-tested
- ✅ Fast for sparse problems
- ✅ Multiple solver options (spsolve, lsqr, etc.)
- ❌ CPU-only
- ❌ Requires tensor ↔ numpy conversion

### 4. Renderer Module (`renderer.py`)

**Purpose:** Convert curves to raster images via rasterization and diffusion.

**Design Rationale:**
- Rasterization places curve colors in image
- Diffusion spreads colors to fill regions
- Two-stage process separates concerns

**Key Functions:**

```python
def rasterize_curve(curve_points, colors, image_size, blur_radius=2.0):
    """
    Design choice: Bresenham-like line drawing.

    Why this approach:
    - Simple and fast
    - Produces connected pixels
    - Easy to understand and debug

    Limitations:
    - No anti-aliasing
    - No sub-pixel accuracy
    - Ignores blur_radius parameter (future work)

    Future enhancements:
    - Xiaolin Wu's line algorithm (anti-aliased)
    - Gaussian splatting for blur
    - GPU-accelerated rasterization
    """
```

```python
def apply_diffusion(image, num_iterations=100, dt=0.1):
    """
    Design choice: Iterative Laplacian diffusion.

    Mathematical model:
    ∂I/∂t = ∇²I (heat equation)

    Discretization:
    I(t+1) = I(t) + dt * Laplacian(I(t))

    Why iterative:
    - Simple to implement
    - Easy to control (num_iterations)
    - Stable with small dt

    Alternative approaches:
    - Direct Poisson solve (faster but more complex)
    - Bilateral filtering (edge-preserving)
    - Anisotropic diffusion (feature-preserving)
    """
```

**Trade-offs:**
- ✅ Simple and understandable
- ✅ Stable and predictable
- ❌ Slow for many iterations
- ❌ No edge preservation
- ❌ May over-blur sharp features

### 5. Pipeline Module (`pipeline.py`)

**Purpose:** Orchestrate all modules into end-to-end vectorization.

**Design Rationale:**
- Single entry point for users
- Encapsulates complexity
- Configurable via constructor parameters

```python
class VectorizationPipeline:
    """
    Design pattern: Facade pattern.

    Responsibilities:
    1. Module coordination
    2. Data flow management
    3. Configuration handling
    4. Result packaging

    Design choices:
    - Solver selection at construction time
    - Immutable configuration (no runtime changes)
    - Returns dict for extensibility
    """

    def __init__(self, solver_type='torch', num_levels=3):
        """
        Configuration philosophy:
        - Sensible defaults for most use cases
        - Explicit parameters for key choices
        - No magic numbers in code
        """

    def vectorize(self, image: torch.Tensor) -> Dict:
        """
        Processing pipeline:

        1. Build pyramids (coarse-to-fine)
        2. For each level:
           a. Detect edges in Laplacian
           b. Trace curves from edges
           c. Solve for curve weights
           d. Create DiffusionCurve objects
        3. Render all curves:
           a. Rasterize each curve
           b. Accumulate in output image
           c. Apply diffusion
        4. Return curves + reconstruction

        Design choice: Process all levels, then render.
        Alternative: Render incrementally (coarse-to-fine).
        """
```

**Trade-offs:**
- ✅ Simple to use
- ✅ Hides complexity
- ✅ Easy to test
- ❌ Less flexible than using modules directly
- ❌ All-or-nothing processing (no partial results)

## Design Patterns Used

### 1. Strategy Pattern (Solvers)
- **Intent:** Define family of algorithms, make them interchangeable
- **Implementation:** `CurveSolver` abstract base class
- **Benefit:** Easy to add new solvers without changing pipeline

### 2. Facade Pattern (Pipeline)
- **Intent:** Provide unified interface to subsystem
- **Implementation:** `VectorizationPipeline` class
- **Benefit:** Simplifies usage, hides complexity

### 3. Data Class Pattern (Data Structures)
- **Intent:** Bundle related data with minimal behavior
- **Implementation:** `@dataclass` for `DiffusionCurve`
- **Benefit:** Clear, concise, type-safe

## Extensibility Points

### Adding a New Solver

```python
# 1. Create new solver class
from .base import CurveSolver

class MySolver(CurveSolver):
    def solve_weights(self, curves, target_laplacian, image_size):
        # Your implementation here
        pass

# 2. Register in __init__.py
from .my_solver import MySolver
__all__ = ['CurveSolver', 'TorchSolver', 'ScipySolver', 'MySolver']

# 3. Update pipeline.py
if solver_type == 'my_solver':
    self.solver = MySolver()
```

### Adding a New Edge Detector

```python
# In curve_extraction.py
def detect_edges_canny(laplacian, low_threshold=0.1, high_threshold=0.3):
    """Alternative edge detector using Canny algorithm"""
    # Implementation here
    pass

# In pipeline.py
def __init__(self, solver_type='torch', num_levels=3, edge_detector='sobel'):
    self.edge_detector = edge_detector

def vectorize(self, image):
    if self.edge_detector == 'sobel':
        edges = detect_edges(laplacian)
    elif self.edge_detector == 'canny':
        edges = detect_edges_canny(laplacian)
```

### Adding Curve Simplification

```python
# In curve_extraction.py
def simplify_curve(curve: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    """Douglas-Peucker curve simplification"""
    # Implementation here
    pass

# In pipeline.py
curves = trace_curves(edges, min_length=10)
curves = [simplify_curve(c, epsilon=2.0) for c in curves]
```

## Performance Considerations

### Memory Usage

**Pyramid Storage:**
- Gaussian pyramid: O(4/3 * H * W * C) for 4 levels
- Laplacian pyramid: Same as Gaussian
- Total: ~2.67x input image size

**Curve Storage:**
- Depends on image complexity
- Typical: 100-1000 curves
- Each curve: N points × (2 coords + 6 colors + 2 blur) = 10N floats
- Total: Usually < 1MB for typical images

### Computational Complexity

**Per Module:**
- Pyramid construction: O(H * W) per level
- Edge detection: O(H * W) per level
- Curve tracing: O(E) where E = number of edge pixels
- Weight solving: O(N) where N = number of curve points (current implementation)
- Rasterization: O(N) per curve
- Diffusion: O(H * W * iterations)

**Bottlenecks:**
1. Diffusion (most expensive)
2. Curve tracing (for complex images)
3. Rasterization (for many curves)

### Optimization Opportunities

1. **Parallel curve processing:**
   ```python
   # Current: Sequential
   for curve in curves:
       render_curve(curve)

   # Better: Parallel
   with torch.multiprocessing.Pool() as pool:
       pool.map(render_curve, curves)
   ```

2. **GPU-accelerated diffusion:**
   ```python
   # Current: Iterative CPU/GPU
   for _ in range(iterations):
       laplacian = conv2d(image, kernel)
       image = image + dt * laplacian

   # Better: Batched GPU operations
   # Or: Direct Poisson solve with sparse GPU solver
   ```

3. **Curve caching:**
   ```python
   # Cache rasterized curves for reuse
   @lru_cache(maxsize=1000)
   def rasterize_curve_cached(curve_hash, ...):
       return rasterize_curve(...)
   ```

## Testing Strategy

### Unit Tests
- Each module tested independently
- Mock dependencies where needed
- Test edge cases (empty inputs, single pixels, etc.)

### Integration Tests
- Full pipeline with both solvers
- Verify output shapes and types
- Check numerical stability

### Comparison Tests
- Benchmark PyTorch vs SciPy
- Verify both produce similar results
- Measure performance characteristics

### Property-Based Tests (Future)
- Use hypothesis for random inputs
- Verify invariants (e.g., reconstruction shape matches input)
- Stress test with extreme parameters

## Common Pitfalls

### 1. Coordinate Systems
```python
# Image: (C, H, W) - channel first
# Curve points: (N, 2) - (x, y) where x is width, y is height
# Be careful: image[y, x] not image[x, y]
```

### 2. Device Mismatches
```python
# Always ensure tensors on same device
curve = curve.to(self.device)
target = target.to(self.device)
```

### 3. Empty Curve Lists
```python
# Always check before processing
if len(curves) > 0:
    weights = solver.solve_weights(curves, ...)
```

### 4. Threshold Sensitivity
```python
# Edge detection threshold affects curve count
# Too low: Too many noisy curves
# Too high: Miss important edges
# Solution: Adaptive thresholding or multi-scale
```

## Future Architecture Improvements

1. **Streaming Pipeline:** Process levels incrementally to reduce memory
2. **Learned Components:** Replace hand-crafted modules with neural networks
3. **GPU Kernels:** Custom CUDA kernels for critical operations
4. **Distributed Processing:** Multi-GPU support for large images
5. **Interactive Editing:** Real-time curve manipulation and re-rendering

## References

- Original paper: "Hierarchical Diffusion Curves for Accurate Automatic Image Vectorization"
- PyTorch documentation: https://pytorch.org/docs/
- SciPy sparse documentation: https://docs.scipy.org/doc/scipy/reference/sparse.html
