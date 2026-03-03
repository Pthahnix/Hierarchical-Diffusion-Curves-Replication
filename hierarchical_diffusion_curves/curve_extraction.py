import torch
import torch.nn.functional as F

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
