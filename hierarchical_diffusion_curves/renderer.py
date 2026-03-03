import torch
import torch.nn.functional as F

def rasterize_curve(
    curve_points: torch.Tensor,
    colors: torch.Tensor,
    image_size: tuple,
    blur_radius: float = 2.0
) -> torch.Tensor:
    """Rasterize a curve with colors to an image

    Args:
        curve_points: Curve control points (N, 2)
        colors: Colors at each point (N, 3)
        image_size: (height, width)
        blur_radius: Blur radius for curve

    Returns:
        Rendered image (3, H, W)
    """
    h, w = image_size
    device = curve_points.device
    image = torch.zeros(3, h, w, device=device)

    # Simple rasterization: draw line segments
    for i in range(len(curve_points) - 1):
        p1 = curve_points[i]
        p2 = curve_points[i + 1]
        c1 = colors[i]
        c2 = colors[i + 1]

        # Bresenham-like line drawing
        x1, y1 = int(p1[0].item()), int(p1[1].item())
        x2, y2 = int(p2[0].item()), int(p2[1].item())

        steps = max(abs(x2 - x1), abs(y2 - y1)) + 1
        for t in range(steps):
            alpha = t / max(steps - 1, 1)
            x = int(x1 + alpha * (x2 - x1))
            y = int(y1 + alpha * (y2 - y1))
            color = c1 * (1 - alpha) + c2 * alpha

            if 0 <= y < h and 0 <= x < w:
                image[:, y, x] = color

    return image

def apply_diffusion(
    image: torch.Tensor,
    num_iterations: int = 100,
    dt: float = 0.1
) -> torch.Tensor:
    """Apply Laplacian diffusion to spread colors

    Args:
        image: Input image (C, H, W)
        num_iterations: Number of diffusion iterations
        dt: Time step for diffusion

    Returns:
        Diffused image (C, H, W)
    """
    result = image.clone()

    # Laplacian kernel
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=image.dtype, device=image.device).view(1, 1, 3, 3)

    for _ in range(num_iterations):
        # Apply Laplacian to each channel
        laplacian = torch.zeros_like(result)
        for c in range(result.shape[0]):
            channel = result[c:c+1].unsqueeze(0)
            lap = F.conv2d(channel, laplacian_kernel, padding=1)
            laplacian[c] = lap[0, 0]

        # Update: I(t+1) = I(t) + dt * Laplacian(I)
        result = result + dt * laplacian

    return result
