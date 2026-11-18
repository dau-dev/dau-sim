__version__ = "0.3.1"

__all__ = (
    "SimulationResult",
    "Simulator",
    "__version__",
)


def __getattr__(name: str):
    # Loaded on first use: the simulator pulls in csp (about 1 s), and hydra
    # imports this package just to find dau_sim.config, so an eager import
    # charged every process that composes the dau-core palette.
    if name in ("SimulationResult", "Simulator"):
        from dau_sim import api

        return getattr(api, name)
    raise AttributeError(f"module 'dau_sim' has no attribute {name!r}")
