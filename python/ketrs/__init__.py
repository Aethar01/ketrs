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
    Charge,
    Circuit,
    DensityMatrix,
    MPS,
    NoiseModel,
    Eigenstate,
    Parameter,
    PauliSum,
    PeriodicMPS,
    SectorBasis,
    SectorStatevector,
    SparseOperator,
    Statevector,
    __version__,
    from_qasm,
    ghz,
    gpu_device,
    lowest_eigenstates,
    qft,
)

__all__ = [
    "Channel",
    "Charge",
    "Circuit",
    "DensityMatrix",
    "MPS",
    "NoiseModel",
    "Eigenstate",
    "Parameter",
    "PauliSum",
    "PeriodicMPS",
    "SectorBasis",
    "SectorStatevector",
    "SparseOperator",
    "Statevector",
    "__version__",
    "from_qasm",
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
