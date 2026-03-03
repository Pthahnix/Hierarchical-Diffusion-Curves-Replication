import torch
import torch.nn.functional as F
from typing import List
import numpy as np

def detect_edges(laplacian: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
    """Detect edges in Laplacian image using gradient magnitude

    Args:
        laplacian: Laplacian image (B, C, H, W)
        threshold: Edge detection threshold

    Returns:
        Binary edge map (B, 1, H, W)
    """
    # Convert to grayscale if needed
    if laplacian.shape[1] == 3:
        gray = 0.299 * laplacian[:, 0] + 0.587 * laplacian[:, 1] + 0.114 * laplacian[:, 2]
        gray = gray.unsqueeze(1)
    else:
        gray = laplacian

    # Sobel filters for gradient
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)

    grad_x = F.conv2d(gray, sobel_x, padding=1)
    grad_y = F.conv2d(gray, sobel_y, padding=1)

    # Gradient magnitude
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

    # Threshold
    edges = (magnitude > threshold).float()

    return edges

def trace_curves(edges: torch.Tensor, min_length: int = 10) -> List[torch.Tensor]:
    """Trace continuous curves from edge map

    Args:
        edges: Binary edge map (B, 1, H, W)
        min_length: Minimum curve length in pixels

    Returns:
        List of curves, each as (N, 2) tensor of coordinates
    """
    edges_np = edges[0, 0].cpu().numpy()
    h, w = edges_np.shape

    visited = np.zeros_like(edges_np, dtype=bool)
    curves = []

    # 8-connected neighbors
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for y in range(h):
        for x in range(w):
            if edges_np[y, x] > 0.5 and not visited[y, x]:
                # Start new curve
                curve = []
                stack = [(y, x)]

                while stack:
                    cy, cx = stack.pop()
                    if visited[cy, cx]:
                        continue

                    visited[cy, cx] = True
                    curve.append([cx, cy])

                    # Check neighbors
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w:
                            if edges_np[ny, nx] > 0.5 and not visited[ny, nx]:
                                stack.append((ny, nx))

                if len(curve) >= min_length:
                    curve_tensor = torch.tensor(curve, dtype=torch.float32, device=edges.device)
                    curves.append(curve_tensor)

    return curves
