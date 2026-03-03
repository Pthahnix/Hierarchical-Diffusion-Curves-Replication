from dataclasses import dataclass
from typing import List
import torch

@dataclass
class DiffusionCurve:
    """Represents a single diffusion curve with color and blur parameters"""
    points: torch.Tensor          # (N, 2) curve control points
    colors_left: torch.Tensor     # (N, 3) left-side RGB colors
    colors_right: torch.Tensor    # (N, 3) right-side RGB colors
    blur_left: torch.Tensor       # (N,) left blur radius
    blur_right: torch.Tensor      # (N,) right blur radius
    scale_level: int              # pyramid level this curve belongs to

@dataclass
class HierarchicalCurves:
    """Container for multi-scale diffusion curves and image pyramids"""
    curves: List[List[DiffusionCurve]]      # [level][curve_idx]
    image_pyramid: List[torch.Tensor]       # Gaussian pyramid
    laplacian_pyramid: List[torch.Tensor]   # Laplacian pyramid
