import torch
from typing import Dict, List
from .prefilter import build_gaussian_pyramid, build_laplacian_pyramid
from .curve_extraction import detect_edges, trace_curves
from .solvers import TorchSolver, ScipySolver
from .renderer import rasterize_curve, apply_diffusion
from .data_structures import DiffusionCurve

class VectorizationPipeline:
    """End-to-end pipeline for image vectorization"""

    def __init__(self, solver_type='torch', num_levels=3):
        self.num_levels = num_levels

        if solver_type == 'torch':
            self.solver = TorchSolver()
        elif solver_type == 'scipy':
            self.solver = ScipySolver()
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

    def vectorize(self, image: torch.Tensor) -> Dict:
        """Vectorize input image to diffusion curves

        Args:
            image: Input image (B, C, H, W)

        Returns:
            Dictionary with 'curves' and 'reconstruction'
        """
        b, c, h, w = image.shape

        # Step 1: Build pyramids
        gaussian_pyr = build_gaussian_pyramid(image, self.num_levels)
        laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)

        # Step 2: Extract curves at each level
        all_curves = []
        for level, laplacian in enumerate(laplacian_pyr[:-1]):
            edges = detect_edges(laplacian, threshold=0.1)
            curves = trace_curves(edges, min_length=10)

            # Solve for weights
            if len(curves) > 0:
                weights = self.solver.solve_weights(
                    curves,
                    laplacian[0],
                    laplacian.shape[2:]
                )

                for curve_pts, curve_weights in zip(curves, weights):
                    all_curves.append(DiffusionCurve(
                        points=curve_pts,
                        colors_left=curve_weights,
                        colors_right=curve_weights,
                        blur_left=torch.ones(len(curve_pts)),
                        blur_right=torch.ones(len(curve_pts)),
                        scale_level=level
                    ))

        # Step 3: Render reconstruction
        reconstruction = torch.zeros(c, h, w, device=image.device)
        for curve in all_curves:
            curve_img = rasterize_curve(
                curve.points,
                curve.colors_left,
                (h, w)
            )
            reconstruction += curve_img

        # Apply diffusion
        reconstruction = apply_diffusion(reconstruction, num_iterations=50)
        reconstruction = reconstruction.unsqueeze(0)

        return {
            'curves': all_curves,
            'reconstruction': reconstruction
        }
