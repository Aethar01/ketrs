"""Quantum computer simulation backend written in Rust.

Qubit ordering follows Qiskit: qubit 0 is the least significant bit of a
basis-state index.

>>> import ketrs
>>> bell = ketrs.Circuit(2).h(0).cx(0, 1)
>>> ketrs.Statevector.from_circuit(bell).probabilities().round(3)
array([0.5, 0. , 0. , 0.5])
"""

from ketrs._ketrs import (
    BosonicMps,
    Channel,
    Charge,
    Circuit,
    CvChannel,
    CvPlan,
    DensityMatrix,
    Eigenstate,
    FixedNFockState,
    FockDensityMatrix,
    FockState,
    GaussianMixture,
    GaussianState,
    HybridCircuit,
    HybridState,
    MPS,
    Measured,
    ModeCircuit,
    ModeState,
    NoiseModel,
    Parameter,
    PauliSum,
    PeriodicMPS,
    SectorBasis,
    SectorStatevector,
    SparseOperator,
    Statevector,
    WindowSampler,
    __version__,
    boson_sample,
    from_qasm,
    ghz,
    gpu_device,
    hafnian,
    hafnian_repeated,
    loop_hafnian,
    lowest_eigenstates,
    permanent,
    plan_circuit,
    qft,
    sample_gbs_pnr,
    sample_gbs_threshold,
    torontonian,
)

__all__ = [
    "BosonicMps",
    "Channel",
    "Charge",
    "Circuit",
    "CvChannel",
    "CvPlan",
    "DensityMatrix",
    "Eigenstate",
    "FixedNFockState",
    "FockDensityMatrix",
    "FockState",
    "GaussianMixture",
    "GaussianState",
    "HybridCircuit",
    "HybridState",
    "MPS",
    "Measured",
    "ModeCircuit",
    "ModeState",
    "NoiseModel",
    "Parameter",
    "PauliSum",
    "PeriodicMPS",
    "SectorBasis",
    "SectorStatevector",
    "SparseOperator",
    "Statevector",
    "WindowSampler",
    "__version__",
    "boson_sample",
    "from_qasm",
    "ghz",
    "gpu_device",
    "hafnian",
    "hafnian_repeated",
    "loop_hafnian",
    "lowest_eigenstates",
    "permanent",
    "plan_circuit",
    "qft",
    "sample_gbs_pnr",
    "sample_gbs_threshold",
    "torontonian",
]

# Present only in builds with GPU support (the `gpu` feature).
try:
    from ketrs._ketrs import GpuFockState, GpuStatevector, permanent_gpu  # noqa: F401

    __all__.extend(["GpuFockState", "GpuStatevector", "permanent_gpu"])
except ImportError:
    pass
