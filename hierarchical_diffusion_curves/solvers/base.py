from abc import ABC, abstractmethod
import torch
from typing import List

class CurveSolver(ABC):
    """Abstract base class for curve weight solvers"""

    @abstractmethod
    def solve_weights(
        self,
        curves: List[torch.Tensor],
        target_laplacian: torch.Tensor,
        image_size: tuple
    ) -> List[torch.Tensor]:
        """Solve for curve color weights to approximate target Laplacian

        Args:
            curves: List of curve point tensors (N, 2)
            target_laplacian: Target Laplacian image (C, H, W)
            image_size: (height, width) of output image

        Returns:
            List of color weight tensors for each curve
        """
        pass
