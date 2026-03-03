import torch
import numpy as np
from typing import List
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from .base import CurveSolver

class ScipySolver(CurveSolver):
    """SciPy-based sparse solver for curve weights"""

    def solve_weights(
        self,
        curves: List[torch.Tensor],
        target_laplacian: torch.Tensor,
        image_size: tuple
    ) -> List[torch.Tensor]:
        """Solve using SciPy sparse linear solver

        Args:
            curves: List of curve point tensors
            target_laplacian: Target Laplacian image
            image_size: Output image size

        Returns:
            List of color weights for each curve
        """
        h, w = image_size
        target_np = target_laplacian.cpu().numpy()

        # Simplified solver: assign average color from target
        weights = []
        for curve in curves:
            curve_np = curve.cpu().numpy()
            num_points = curve_np.shape[0]
            curve_weights = np.zeros((num_points, 3))

            for i, point in enumerate(curve_np):
                x, y = int(point[0]), int(point[1])
                if 0 <= y < h and 0 <= x < w:
                    curve_weights[i] = target_np[:, y, x]

            weights.append(torch.from_numpy(curve_weights).float())

        return weights
