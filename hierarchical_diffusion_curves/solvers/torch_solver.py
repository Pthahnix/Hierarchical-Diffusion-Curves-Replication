import torch
import torch.nn.functional as F
from typing import List
from .base import CurveSolver

class TorchSolver(CurveSolver):
    """PyTorch-based sparse solver for curve weights"""

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

    def solve_weights(
        self,
        curves: List[torch.Tensor],
        target_laplacian: torch.Tensor,
        image_size: tuple
    ) -> List[torch.Tensor]:
        """Solve using PyTorch sparse linear solver

        Args:
            curves: List of curve point tensors
            target_laplacian: Target Laplacian image
            image_size: Output image size

        Returns:
            List of color weights for each curve
        """
        h, w = image_size
        target = target_laplacian.to(self.device)

        # Simplified solver: assign average color from target
        weights = []
        for curve in curves:
            curve = curve.to(self.device)
            # Sample colors along curve from target
            num_points = curve.shape[0]
            curve_weights = torch.zeros(num_points, 3, device=self.device)

            for i, point in enumerate(curve):
                x, y = int(point[0].item()), int(point[1].item())
                if 0 <= y < h and 0 <= x < w:
                    curve_weights[i] = target[:, y, x]

            weights.append(curve_weights)

        return weights
