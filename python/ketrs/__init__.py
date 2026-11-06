"""Quantum computer simulation backend written in Rust.

Qubit ordering follows Qiskit: qubit 0 is the least significant bit of a
basis-state index.

>>> import ketrs
>>> bell = ketrs.Circuit(2).h(0).cx(0, 1)
>>> ketrs.Statevector.from_circuit(bell).probabilities().round(3)
array([0.5, 0. , 0. , 0.5])
"""

from ketrs._ketrs import (
    Channel,
    Circuit,
    DensityMatrix,
    MPS,
    NoiseModel,
    Eigenstate,
    Parameter,
    PauliSum,
    SectorBasis,
    SectorStatevector,
    Statevector,
    __version__,
    ghz,
    gpu_device,
    lowest_eigenstates,
    qft,
)

__all__ = [
    "Channel",
    "Circuit",
    "DensityMatrix",
    "MPS",
    "NoiseModel",
    "Eigenstate",
    "Parameter",
    "PauliSum",
    "SectorBasis",
    "SectorStatevector",
    "Statevector",
    "__version__",
    "ghz",
    "gpu_device",
    "lowest_eigenstates",
    "qft",
]

# Present only in builds with GPU support (the `gpu` feature).
try:
    from ketrs._ketrs import GpuStatevector  # noqa: F401

    __all__.append("GpuStatevector")
except ImportError:
    pass
