from __future__ import annotations

__all__ = ("from_amaranth", "from_dau_build", "parse_sv", "parse_sv_file")


def __getattr__(name: str):
    if name == "from_amaranth":
        from dau_sim.frontends.amaranth_frontend import from_amaranth

        return from_amaranth
    if name in {"from_dau_build", "parse_sv", "parse_sv_file"}:
        from dau_sim.frontends import pyslang_frontend

        return getattr(pyslang_frontend, name)
    raise AttributeError(name)
