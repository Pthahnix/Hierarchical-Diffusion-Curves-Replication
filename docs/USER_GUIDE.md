# User Guide

## Getting Started

### Installation

#### Prerequisites
- Python 3.11 or higher
- pip package manager

#### Install from source
```bash
# Clone or navigate to project directory
cd hierarchical_diffusion_curves

# Install in development mode
pip install -e .

# Or install normally
pip install .
```

#### Verify installation
```bash
python -c "from hierarchical_diffusion_curves.pipeline import VectorizationPipeline; print('✓ Installation successful')"
```

## Basic Usage

### Python API

#### Minimal Example
```python
import torch
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline

# Create pipeline
pipeline = VectorizationPipeline(solver_type='torch', num_levels=3)

# Load or create image (B, C, H, W format)
image = torch.rand(1, 3, 256, 256)

# Vectorize
result = pipeline.vectorize(image)

# Access results
curves = result['curves']
reconstruction = result['reconstruction']

print(f"Found {len(curves)} curves")
print(f"Reconstruction shape: {reconstruction.shape}")
```

#### Complete Example with Image I/O
```python
import torch
from PIL import Image
import torchvision.transforms as T
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline

# Load image from file
image_path = "input.jpg"
image = Image.open(image_path).convert('RGB')

# Convert to tensor
transform = T.Compose([
    T.Resize((256, 256)),  # Optional: resize for faster processing
    T.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

# Vectorize
pipeline = VectorizationPipeline(solver_type='torch', num_levels=3)
result = pipeline.vectorize(image_tensor)

# Save reconstruction
reconstruction = result['reconstruction'][0]  # Remove batch dimension
reconstruction = reconstruction.clamp(0, 1)  # Ensure valid range
output_image = T.ToPILImage()(reconstruction)
output_image.save("output.jpg")

print(f"Vectorization complete!")
print(f"Found {len(result['curves'])} curves")
```

### Command Line Interface

#### Basic Usage
```bash
python examples/vectorize_image.py input.jpg output.jpg
```

#### With Options
```bash
# Use SciPy solver
python examples/vectorize_image.py input.jpg output.jpg --solver scipy

# Change pyramid levels
python examples/vectorize_image.py input.jpg output.jpg --levels 4

# Combine options
python examples/vectorize_image.py input.jpg output.jpg --solver torch --levels 3
```

#### Help
```bash
python examples/vectorize_image.py --help
```

## Configuration Options

### Pipeline Parameters

#### `solver_type` (str)
Chooses which solver backend to use.

**Options:**
- `'torch'` - PyTorch-based solver (GPU-capable)
- `'scipy'` - SciPy sparse solver (CPU-based)

**When to use:**
- Use `'torch'` for large images (>512x512) if GPU available
- Use `'scipy'` for small images or CPU-only systems
- Use `'scipy'` for maximum stability and accuracy

**Example:**
```python
# GPU-accelerated
pipeline = VectorizationPipeline(solver_type='torch')

# CPU-based, stable
pipeline = VectorizationPipeline(solver_type='scipy')
```

#### `num_levels` (int)
Number of pyramid levels for multi-scale processing.

**Range:** 2-5 (typical)

**Trade-offs:**
- More levels: Better handling of different scales, slower
- Fewer levels: Faster, may miss coarse structures

**Guidelines:**
- Small images (128x128): 2-3 levels
- Medium images (256x256): 3-4 levels
- Large images (512x512+): 4-5 levels

**Example:**
```python
# Fast, fewer scales
pipeline = VectorizationPipeline(num_levels=2)

# Thorough, more scales
pipeline = VectorizationPipeline(num_levels=5)
```

## Advanced Usage

### Accessing Curve Data

```python
result = pipeline.vectorize(image)

for i, curve in enumerate(result['curves']):
    print(f"Curve {i}:")
    print(f"  Points: {curve.points.shape}")  # (N, 2) - x, y coordinates
    print(f"  Colors left: {curve.colors_left.shape}")  # (N, 3) - RGB
    print(f"  Colors right: {curve.colors_right.shape}")  # (N, 3) - RGB
    print(f"  Blur left: {curve.blur_left.shape}")  # (N,)
    print(f"  Blur right: {curve.blur_right.shape}")  # (N,)
    print(f"  Scale level: {curve.scale_level}")  # Pyramid level
```

### Comparing Solvers

```python
import time
import torch
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline

# Load image
image = torch.rand(1, 3, 256, 256)

# Compare both solvers
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

    print(f"{solver_type:6s}: {elapsed:.3f}s, {len(result['curves'])} curves")

# Compare reconstructions
torch_recon = results['torch']['reconstruction']
scipy_recon = results['scipy']['reconstruction']
difference = torch.abs(torch_recon - scipy_recon).mean()
print(f"Mean absolute difference: {difference:.6f}")
```

### Batch Processing

```python
import os
from pathlib import Path
from PIL import Image
import torchvision.transforms as T
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline

# Setup
input_dir = Path("input_images")
output_dir = Path("output_images")
output_dir.mkdir(exist_ok=True)

pipeline = VectorizationPipeline(solver_type='torch', num_levels=3)
transform = T.Compose([T.ToTensor()])

# Process all images
for img_path in input_dir.glob("*.jpg"):
    print(f"Processing {img_path.name}...")

    # Load and vectorize
    image = Image.open(img_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    result = pipeline.vectorize(image_tensor)

    # Save reconstruction
    reconstruction = result['reconstruction'][0].clamp(0, 1)
    output_image = T.ToPILImage()(reconstruction)
    output_path = output_dir / img_path.name
    output_image.save(output_path)

    print(f"  → Saved to {output_path} ({len(result['curves'])} curves)")
```

### Custom Processing Pipeline

```python
from hierarchical_diffusion_curves.prefilter import build_gaussian_pyramid, build_laplacian_pyramid
from hierarchical_diffusion_curves.curve_extraction import detect_edges, trace_curves
from hierarchical_diffusion_curves.solvers import TorchSolver
from hierarchical_diffusion_curves.renderer import rasterize_curve, apply_diffusion

# Build pyramids
gaussian_pyr = build_gaussian_pyramid(image, num_levels=3)
laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)

# Process specific level
level = 1
laplacian = laplacian_pyr[level]

# Extract curves with custom threshold
edges = detect_edges(laplacian, threshold=0.15)  # Custom threshold
curves = trace_curves(edges, min_length=20)  # Longer minimum length

# Solve weights
solver = TorchSolver()
weights = solver.solve_weights(curves, laplacian[0], laplacian.shape[2:])

# Render with custom parameters
h, w = image.shape[2:]
reconstruction = torch.zeros(3, h, w)
for curve_pts, curve_weights in zip(curves, weights):
    curve_img = rasterize_curve(curve_pts, curve_weights, (h, w))
    reconstruction += curve_img

# Apply more diffusion iterations
reconstruction = apply_diffusion(reconstruction, num_iterations=200)
```

## Performance Tips

### GPU Acceleration

```python
# Check GPU availability
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# Move image to GPU
if torch.cuda.is_available():
    image = image.cuda()

# PyTorch solver will automatically use GPU
pipeline = VectorizationPipeline(solver_type='torch')
result = pipeline.vectorize(image)
```

### Memory Management

```python
import torch

# For large images, process in smaller chunks or reduce levels
def vectorize_large_image(image_path, max_size=512):
    from PIL import Image
    import torchvision.transforms as T

    # Load and resize if needed
    image = Image.open(image_path).convert('RGB')
    w, h = image.size

    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.LANCZOS)
        print(f"Resized from {w}x{h} to {new_size[0]}x{new_size[1]}")

    # Process
    transform = T.ToTensor()
    image_tensor = transform(image).unsqueeze(0)

    pipeline = VectorizationPipeline(solver_type='torch', num_levels=3)
    result = pipeline.vectorize(image_tensor)

    return result

# Clear GPU cache after processing
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

### Choosing Parameters for Speed

```python
# Fast mode (lower quality)
pipeline = VectorizationPipeline(
    solver_type='scipy',  # Faster for small images
    num_levels=2          # Fewer levels
)

# Quality mode (slower)
pipeline = VectorizationPipeline(
    solver_type='torch',  # GPU acceleration
    num_levels=4          # More scales
)
```

## Troubleshooting

### Common Issues

#### Import Error
```
ModuleNotFoundError: No module named 'hierarchical_diffusion_curves'
```

**Solution:**
```bash
pip install -e .
```

#### CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```python
# 1. Use CPU solver
pipeline = VectorizationPipeline(solver_type='scipy')

# 2. Reduce image size
image = T.Resize((256, 256))(image)

# 3. Reduce pyramid levels
pipeline = VectorizationPipeline(num_levels=2)

# 4. Clear cache
torch.cuda.empty_cache()
```

#### No Curves Found
```python
result = pipeline.vectorize(image)
print(len(result['curves']))  # 0
```

**Causes:**
- Image too smooth (no edges)
- Threshold too high
- Image too small

**Solutions:**
```python
# Lower edge detection threshold (modify source)
# Or use more contrasted input image
# Or increase image size
```

#### Poor Reconstruction Quality

**Possible causes:**
- Too few pyramid levels
- Edge detection threshold not optimal
- Insufficient diffusion iterations

**Solutions:**
```python
# Increase pyramid levels
pipeline = VectorizationPipeline(num_levels=4)

# Modify diffusion iterations (in source)
# In pipeline.py, change:
# reconstruction = apply_diffusion(reconstruction, num_iterations=100)
```

## Examples Gallery

### Example 1: Simple Geometric Shape
```python
import torch
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline

# Create test image: white square on black background
image = torch.zeros(1, 3, 128, 128)
image[:, :, 40:88, 40:88] = 1.0

pipeline = VectorizationPipeline(solver_type='torch', num_levels=3)
result = pipeline.vectorize(image)

print(f"Curves found: {len(result['curves'])}")
# Expected: 4 curves (one per edge of square)
```

### Example 2: Gradient Image
```python
import torch
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline

# Create gradient image
image = torch.zeros(1, 3, 128, 128)
for i in range(128):
    image[:, :, i, :] = i / 127.0

pipeline = VectorizationPipeline(solver_type='torch', num_levels=3)
result = pipeline.vectorize(image)

print(f"Curves found: {len(result['curves'])}")
# Expected: Few or no curves (smooth gradient has no sharp edges)
```

### Example 3: Real Photo
```python
from PIL import Image
import torchvision.transforms as T
from hierarchical_diffusion_curves.pipeline import VectorizationPipeline

# Load photo
image = Image.open("photo.jpg").convert('RGB')
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])
image_tensor = transform(image).unsqueeze(0)

# Vectorize
pipeline = VectorizationPipeline(solver_type='torch', num_levels=4)
result = pipeline.vectorize(image_tensor)

# Save results
reconstruction = result['reconstruction'][0].clamp(0, 1)
output = T.ToPILImage()(reconstruction)
output.save("photo_vectorized.jpg")

print(f"Curves found: {len(result['curves'])}")
# Expected: Many curves (100-1000+ depending on photo complexity)
```

## Best Practices

### 1. Image Preprocessing
```python
# Normalize image to [0, 1] range
image = image / 255.0 if image.max() > 1.0 else image

# Ensure RGB (not RGBA)
if image.shape[1] == 4:
    image = image[:, :3, :, :]

# Resize for consistent processing
image = T.Resize((256, 256))(image)
```

### 2. Result Validation
```python
result = pipeline.vectorize(image)

# Check for valid output
assert 'curves' in result
assert 'reconstruction' in result
assert result['reconstruction'].shape == image.shape

# Check reconstruction range
recon = result['reconstruction']
print(f"Reconstruction range: [{recon.min():.3f}, {recon.max():.3f}]")
```

### 3. Error Handling
```python
try:
    result = pipeline.vectorize(image)
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print("GPU memory error, falling back to CPU solver")
        pipeline = VectorizationPipeline(solver_type='scipy')
        result = pipeline.vectorize(image.cpu())
    else:
        raise
```

## API Reference

See `CLAUDE.md` for complete API documentation.

### Key Classes

- `VectorizationPipeline` - Main entry point
- `DiffusionCurve` - Curve data structure
- `CurveSolver` - Abstract solver interface
- `TorchSolver` - PyTorch implementation
- `ScipySolver` - SciPy implementation

### Key Functions

- `build_gaussian_pyramid()` - Multi-scale decomposition
- `build_laplacian_pyramid()` - Detail extraction
- `detect_edges()` - Edge detection
- `trace_curves()` - Curve extraction
- `rasterize_curve()` - Curve rendering
- `apply_diffusion()` - Color spreading

## Further Reading

- `ARCHITECTURE.md` - Detailed design documentation
- `IMPLEMENTATION_NOTES.md` - Technical implementation details
- `CLAUDE.md` - Complete project memory and reference
- Original paper: "Hierarchical Diffusion Curves for Accurate Automatic Image Vectorization"
