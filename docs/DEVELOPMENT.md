# Development Guide

## Development Philosophy

This project follows Test-Driven Development (TDD) principles with emphasis on:
- **Simplicity:** Minimal code to solve the problem
- **Clarity:** Readable over clever
- **Modularity:** Independent, testable components
- **Documentation:** Code explains "how", docs explain "why"

## Development Setup

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd hierarchical_diffusion_curves

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dependencies
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black flake8 mypy
```

### IDE Configuration

#### VS Code
```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

#### PyCharm
- Enable pytest as test runner
- Configure Black as code formatter
- Enable type checking with mypy

## Development Workflow

### TDD Cycle

This project was built using strict TDD. Follow this cycle for new features:

```
1. Write a failing test
   ↓
2. Run test (verify it fails)
   ↓
3. Write minimal code to pass
   ↓
4. Run test (verify it passes)
   ↓
5. Refactor if needed
   ↓
6. Commit with semantic message
   ↓
7. Repeat
```

### Example: Adding a New Feature

Let's add a curve simplification feature using TDD:

#### Step 1: Write the test
```python
# tests/test_curve_extraction.py

def test_simplify_curve():
    """Test Douglas-Peucker curve simplification"""
    from hierarchical_diffusion_curves.curve_extraction import simplify_curve

    # Create curve with many points
    curve = torch.tensor([
        [0.0, 0.0],
        [1.0, 0.1],  # Nearly collinear
        [2.0, 0.0],
        [3.0, 3.0],
    ])

    # Simplify with epsilon=0.5
    simplified = simplify_curve(curve, epsilon=0.5)

    # Should remove middle collinear point
    assert len(simplified) < len(curve)
    assert simplified[0].tolist() == [0.0, 0.0]
    assert simplified[-1].tolist() == [3.0, 3.0]
```

#### Step 2: Run test (should fail)
```bash
pytest tests/test_curve_extraction.py::test_simplify_curve -v
# Expected: ImportError or AttributeError
```

#### Step 3: Implement minimal code
```python
# hierarchical_diffusion_curves/curve_extraction.py

def simplify_curve(curve: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    """Simplify curve using Douglas-Peucker algorithm

    Args:
        curve: Curve points (N, 2)
        epsilon: Simplification threshold

    Returns:
        Simplified curve (M, 2) where M <= N
    """
    if len(curve) <= 2:
        return curve

    # Find point with maximum distance from line
    start, end = curve[0], curve[-1]
    dists = point_line_distance(curve[1:-1], start, end)
    max_dist, max_idx = dists.max(), dists.argmax()

    # If max distance > epsilon, recursively simplify
    if max_dist > epsilon:
        # Recursively simplify left and right segments
        left = simplify_curve(curve[:max_idx+2], epsilon)
        right = simplify_curve(curve[max_idx+1:], epsilon)
        return torch.cat([left[:-1], right])
    else:
        # All points close to line, keep only endpoints
        return torch.stack([start, end])

def point_line_distance(points: torch.Tensor, start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Calculate perpendicular distance from points to line"""
    # Vector from start to end
    line_vec = end - start
    line_len = torch.norm(line_vec)

    if line_len < 1e-6:
        return torch.norm(points - start, dim=1)

    # Normalized line direction
    line_dir = line_vec / line_len

    # Vector from start to each point
    point_vecs = points - start

    # Project onto line
    projections = torch.sum(point_vecs * line_dir, dim=1, keepdim=True) * line_dir

    # Perpendicular distance
    perpendicular = point_vecs - projections
    distances = torch.norm(perpendicular, dim=1)

    return distances
```

#### Step 4: Run test (should pass)
```bash
pytest tests/test_curve_extraction.py::test_simplify_curve -v
# Expected: PASSED
```

#### Step 5: Refactor if needed
- Check for edge cases
- Optimize if necessary
- Add docstrings
- Ensure type hints

#### Step 6: Commit
```bash
git add hierarchical_diffusion_curves/curve_extraction.py tests/test_curve_extraction.py
git commit -m "feat: add Douglas-Peucker curve simplification"
```

## Testing Guidelines

### Test Structure

```python
def test_feature_name():
    """Clear description of what is being tested"""
    # Arrange: Set up test data
    input_data = create_test_input()

    # Act: Execute the function
    result = function_under_test(input_data)

    # Assert: Verify the result
    assert result.shape == expected_shape
    assert result.dtype == expected_dtype
    assert torch.allclose(result, expected_value)
```

### Test Categories

#### 1. Unit Tests
Test individual functions in isolation.

```python
def test_detect_edges_shape():
    """Test edge detection output shape"""
    laplacian = torch.rand(1, 3, 64, 64)
    edges = detect_edges(laplacian, threshold=0.1)
    assert edges.shape == (1, 1, 64, 64)

def test_detect_edges_range():
    """Test edge detection output range"""
    laplacian = torch.rand(1, 3, 64, 64)
    edges = detect_edges(laplacian, threshold=0.1)
    assert edges.min() >= 0.0
    assert edges.max() <= 1.0
```

#### 2. Integration Tests
Test multiple components working together.

```python
def test_full_pipeline_torch():
    """Test complete pipeline with PyTorch solver"""
    pipeline = VectorizationPipeline(solver_type='torch')
    image = torch.rand(1, 3, 128, 128)
    result = pipeline.vectorize(image)

    assert 'curves' in result
    assert 'reconstruction' in result
    assert result['reconstruction'].shape == image.shape
```

#### 3. Property Tests
Test invariants that should always hold.

```python
def test_pyramid_sizes():
    """Test pyramid levels have correct sizes"""
    image = torch.rand(1, 3, 256, 256)
    pyramid = build_gaussian_pyramid(image, num_levels=4)

    # Each level should be half the size of previous
    for i in range(len(pyramid) - 1):
        h1, w1 = pyramid[i].shape[2:]
        h2, w2 = pyramid[i+1].shape[2:]
        assert h2 == h1 // 2
        assert w2 == w1 // 2
```

#### 4. Regression Tests
Prevent bugs from reappearing.

```python
def test_empty_curve_list_handling():
    """Regression: Pipeline should handle images with no edges"""
    # This used to crash with empty curve list
    pipeline = VectorizationPipeline(solver_type='torch')

    # Completely uniform image (no edges)
    image = torch.ones(1, 3, 64, 64) * 0.5

    result = pipeline.vectorize(image)
    assert len(result['curves']) == 0  # No curves expected
    assert result['reconstruction'].shape == image.shape
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_pipeline.py -v

# Specific test
pytest tests/test_pipeline.py::test_pipeline_basic -v

# With coverage
pytest tests/ --cov=hierarchical_diffusion_curves --cov-report=html

# With output (for debugging)
pytest tests/ -v -s

# Stop on first failure
pytest tests/ -x

# Run only failed tests from last run
pytest tests/ --lf
```

## Code Style

### Formatting

Use Black for consistent formatting:
```bash
# Format all code
black hierarchical_diffusion_curves/ tests/

# Check without modifying
black --check hierarchical_diffusion_curves/ tests/
```

### Linting

Use flake8 for style checking:
```bash
# Check all code
flake8 hierarchical_diffusion_curves/ tests/

# With specific rules
flake8 --max-line-length=100 --ignore=E203,W503 hierarchical_diffusion_curves/
```

### Type Hints

Use type hints for all public functions:
```python
from typing import List, Tuple, Optional
import torch

def process_curves(
    curves: List[torch.Tensor],
    threshold: float = 0.1,
    device: Optional[str] = None
) -> Tuple[List[torch.Tensor], int]:
    """Process curves with type hints

    Args:
        curves: List of curve tensors
        threshold: Processing threshold
        device: Optional device specification

    Returns:
        Tuple of (processed curves, count)
    """
    pass
```

Check types with mypy:
```bash
mypy hierarchical_diffusion_curves/
```

## Git Workflow

### Commit Messages

Follow semantic commit conventions:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions/changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Maintenance tasks

**Examples:**
```bash
git commit -m "feat: add curve simplification algorithm"
git commit -m "fix: handle empty curve lists in renderer"
git commit -m "docs: add architecture guide"
git commit -m "test: add integration tests for scipy solver"
git commit -m "refactor: extract edge detection to separate function"
git commit -m "perf: optimize diffusion with batched operations"
git commit -m "chore: update dependencies"
```

### Branch Strategy

```bash
# Main branch: stable, tested code
main

# Development branch: integration branch
develop

# Feature branches: new features
feature/curve-simplification
feature/bilateral-filtering

# Fix branches: bug fixes
fix/empty-curve-handling
fix/memory-leak

# Create feature branch
git checkout -b feature/my-feature

# Work on feature (TDD cycle)
# ... make changes ...
git add .
git commit -m "feat: add my feature"

# Merge back to develop
git checkout develop
git merge feature/my-feature

# Delete feature branch
git branch -d feature/my-feature
```

## Debugging

### Print Debugging

```python
def debug_pipeline(image):
    """Debug version of pipeline with intermediate outputs"""
    print(f"Input shape: {image.shape}")

    # Pyramids
    gaussian_pyr = build_gaussian_pyramid(image, num_levels=3)
    print(f"Gaussian pyramid levels: {[p.shape for p in gaussian_pyr]}")

    laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)
    print(f"Laplacian pyramid levels: {[p.shape for p in laplacian_pyr]}")

    # Curves
    for level, laplacian in enumerate(laplacian_pyr[:-1]):
        edges = detect_edges(laplacian, threshold=0.1)
        print(f"Level {level} edges: {edges.sum().item()} pixels")

        curves = trace_curves(edges, min_length=10)
        print(f"Level {level} curves: {len(curves)}")

        if len(curves) > 0:
            print(f"  Curve lengths: {[len(c) for c in curves]}")
```

### Visual Debugging

```python
import matplotlib.pyplot as plt

def visualize_pyramids(image):
    """Visualize Gaussian and Laplacian pyramids"""
    gaussian_pyr = build_gaussian_pyramid(image, num_levels=4)
    laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, (g, l) in enumerate(zip(gaussian_pyr, laplacian_pyr)):
        # Gaussian
        axes[0, i].imshow(g[0].permute(1, 2, 0).cpu())
        axes[0, i].set_title(f'Gaussian L{i}')
        axes[0, i].axis('off')

        # Laplacian
        axes[1, i].imshow(l[0].permute(1, 2, 0).cpu())
        axes[1, i].set_title(f'Laplacian L{i}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig('pyramids_debug.png')
    plt.close()

def visualize_curves(image, curves):
    """Visualize extracted curves on image"""
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 10))
    plt.imshow(image[0].permute(1, 2, 0).cpu())

    for curve in curves:
        points = curve.cpu().numpy()
        plt.plot(points[:, 0], points[:, 1], 'r-', linewidth=2)

    plt.title(f'{len(curves)} curves')
    plt.axis('off')
    plt.savefig('curves_debug.png')
    plt.close()
```

### Profiling

```python
import time
import torch

def profile_pipeline():
    """Profile pipeline performance"""
    image = torch.rand(1, 3, 256, 256)
    pipeline = VectorizationPipeline(solver_type='torch', num_levels=3)

    # Warmup
    _ = pipeline.vectorize(image)

    # Profile
    times = {}

    start = time.time()
    gaussian_pyr = build_gaussian_pyramid(image, num_levels=3)
    times['gaussian_pyramid'] = time.time() - start

    start = time.time()
    laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)
    times['laplacian_pyramid'] = time.time() - start

    # ... profile other components ...

    for name, t in times.items():
        print(f"{name:20s}: {t*1000:.2f}ms")
```

## Performance Optimization

### Profiling Tools

```bash
# CPU profiling with cProfile
python -m cProfile -o profile.stats examples/vectorize_image.py input.jpg output.jpg

# Analyze profile
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Memory profiling
pip install memory_profiler
python -m memory_profiler examples/vectorize_image.py input.jpg output.jpg
```

### Optimization Checklist

1. **Profile first** - Don't optimize without data
2. **Vectorize operations** - Use PyTorch batch operations
3. **Minimize data transfer** - Keep tensors on same device
4. **Cache results** - Avoid recomputing same values
5. **Use appropriate dtypes** - float16 for memory, float32 for accuracy
6. **Parallelize** - Use multiprocessing for independent tasks

### Example Optimization

**Before:**
```python
# Slow: Loop over pixels
for y in range(h):
    for x in range(w):
        if edges[y, x] > 0.5:
            process_pixel(y, x)
```

**After:**
```python
# Fast: Vectorized operation
edge_pixels = torch.nonzero(edges > 0.5)
process_pixels_batch(edge_pixels)
```

## Documentation

### Docstring Format

Use Google-style docstrings:

```python
def function_name(arg1: type1, arg2: type2) -> return_type:
    """Short one-line summary.

    Longer description if needed. Explain what the function does,
    not how it does it (code shows how).

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When arg1 is negative
        RuntimeError: When computation fails

    Example:
        >>> result = function_name(1, 2)
        >>> print(result)
        3
    """
    pass
```

### Adding Documentation

```bash
# Update relevant docs when adding features
docs/
├── ARCHITECTURE.md      # Design decisions
├── USER_GUIDE.md        # Usage examples
├── DEVELOPMENT.md       # This file
└── IMPLEMENTATION_NOTES.md  # Technical details

# Also update
CLAUDE.md               # Project memory
README.md               # Quick start
```

## Contributing

### Before Submitting

1. **Run all tests**: `pytest tests/ -v`
2. **Check coverage**: `pytest tests/ --cov=hierarchical_diffusion_curves`
3. **Format code**: `black hierarchical_diffusion_curves/ tests/`
4. **Lint code**: `flake8 hierarchical_diffusion_curves/ tests/`
5. **Type check**: `mypy hierarchical_diffusion_curves/`
6. **Update docs**: Add/update relevant documentation

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] Added tests for new functionality
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] No unnecessary dependencies added
```

## Troubleshooting Development Issues

### Test Failures

```bash
# Run with verbose output
pytest tests/ -v -s

# Run specific failing test
pytest tests/test_file.py::test_name -v -s

# Debug with pdb
pytest tests/test_file.py::test_name --pdb
```

### Import Errors

```bash
# Reinstall in development mode
pip uninstall hierarchical_diffusion_curves
pip install -e .

# Check Python path
python -c "import sys; print('\n'.join(sys.path))"
```

### GPU Issues

```python
# Check CUDA availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device count: {torch.cuda.device_count()}")

# Force CPU mode
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

## Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **SciPy Documentation**: https://docs.scipy.org/
- **pytest Documentation**: https://docs.pytest.org/
- **Black Formatter**: https://black.readthedocs.io/
- **Type Hints (PEP 484)**: https://www.python.org/dev/peps/pep-0484/

## Questions?

See `CLAUDE.md` for comprehensive project documentation or check existing issues/documentation.
