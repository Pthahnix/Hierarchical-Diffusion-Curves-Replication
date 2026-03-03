import torch
from hierarchical_diffusion_curves.solvers.base import CurveSolver
from hierarchical_diffusion_curves.solvers.torch_solver import TorchSolver

def test_solver_interface():
    """Test that solver interface is abstract"""
    try:
        solver = CurveSolver()
        assert False, "Should not be able to instantiate abstract class"
    except TypeError:
        pass  # Expected

def test_torch_solver_basic():
    """Test PyTorch solver on simple problem"""
    solver = TorchSolver()

    # Simple test: single curve
    curves = [torch.tensor([[10.0, 10.0], [20.0, 20.0]])]
    target = torch.zeros(3, 64, 64)
    target[:, 10:21, 10:21] = 1.0

    weights = solver.solve_weights(curves, target, (64, 64))

    assert len(weights) == 1
    assert weights[0].shape[0] == 2  # Two points
