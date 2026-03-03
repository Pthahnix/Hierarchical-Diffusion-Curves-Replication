import torch
from hierarchical_diffusion_curves.solvers.base import CurveSolver

def test_solver_interface():
    """Test that solver interface is abstract"""
    try:
        solver = CurveSolver()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass  # Expected
