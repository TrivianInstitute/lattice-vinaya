# The Trivian Lattice System
# Exports both the Runtime (Interaction) and Core (Governance) layers.

from .runtime import (
    enforce_lattice_vinaya, 
    LatticeLedger, 
    load_vinaya_json
)

from .core import (
    VinayaGovernor, 
    LatticeHealth, 
    InteractionContext,
    Node,
    InteractionResult
)

__version__ = "2.0.0-hybrid"
