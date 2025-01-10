"""
Solvers module.

==================================================================

The module aims to provide methods and classes for configuration of different \
local and global solvers.
"""

# Author: Tim Ortkamp <tim.ortkamp@kit.edu>

try:
    from ._ipopt import IpoptSolver
except ImportError:
    IpoptSolver = None
from ._scipy_solver import SciPySolver

__all__ = ["IpoptSolver", "SciPySolver"]
