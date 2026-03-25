"""Hierarchy flattener.

Takes a hierarchical ``Module`` (with ``instances`` and ``submodules``)
and produces a flat ``Module`` where all child signals, blocks, and
memories have been merged into the top level with dotted name prefixes.
"""

from __future__ import annotations

from dau_sim.compiler.rewrite import prefix_stmts
from dau_sim.ir.expr import SignalRef
from dau_sim.ir.module import (
    ClockDomain,
    CombBlock,
    InitBlock,
    Memory,
    Module,
    ReadPort,
    SeqBlock,
    Signal,
    WritePort,
)
from dau_sim.ir.stmt import Assign
from dau_sim.ir.types import PortDirection


def flatten_module(module: Module) -> Module:
    """Flatten a hierarchical Module into a single flat Module.

    If the module has no instances, it is returned as-is (with
    ``submodules`` cleared for cleanliness).
    """
    if not module.instances and not module.submodules:
        return module

    # Build lookup from module name → Module definition
    mod_defs: dict[str, Module] = {}
    for sm in module.submodules:
        mod_defs[sm.name] = sm

    # Accumulators for the flat module
    ports = list(module.ports)
    signals = list(module.signals)
    clock_domains = list(module.clock_domains)
    comb_blocks = list(module.comb_blocks)
    seq_blocks = list(module.seq_blocks)
    init_blocks = list(module.init_blocks)
    memories = list(module.memories)

    for inst in module.instances:
        child_def = mod_defs.get(inst.module_name)
        if child_def is None:
            raise ValueError(f"Instance '{inst.name}' references module '{inst.module_name}' but no definition found in submodules")

        # Recursively flatten the child first
        child_flat = flatten_module(child_def)

        prefix = inst.name

        # Build port direction lookup for the child
        child_port_dir: dict[str, PortDirection] = {}
        for p in child_flat.ports:
            child_port_dir[p.name] = p.direction

        # Add child signals (prefixed) to parent
        for p in child_flat.ports:
            signals.append(
                Signal(
                    name=f"{prefix}.{p.name}",
                    shape=p.shape,
                    init=p.signal.init,
                    net_kind=p.signal.net_kind,
                )
            )
        for s in child_flat.signals:
            signals.append(
                Signal(
                    name=f"{prefix}.{s.name}",
                    shape=s.shape,
                    init=s.init,
                    net_kind=s.net_kind,
                )
            )

        # Add child clock domains (prefixed)
        for cd in child_flat.clock_domains:
            clock_domains.append(
                ClockDomain(
                    name=f"{prefix}.{cd.name}",
                    clk=f"{prefix}.{cd.clk}",
                    edge=cd.edge,
                    rst=f"{prefix}.{cd.rst}" if cd.rst else None,
                    rst_style=cd.rst_style,
                    rst_active_high=cd.rst_active_high,
                )
            )

        # Add child comb blocks (prefixed)
        for cb in child_flat.comb_blocks:
            comb_blocks.append(CombBlock(stmts=prefix_stmts(cb.stmts, prefix)))

        # Add child seq blocks (prefixed)
        for sb in child_flat.seq_blocks:
            seq_blocks.append(
                SeqBlock(
                    domain=f"{prefix}.{sb.domain}",
                    stmts=prefix_stmts(sb.stmts, prefix),
                )
            )

        # Add child init blocks (prefixed)
        for ib in child_flat.init_blocks:
            init_blocks.append(InitBlock(stmts=prefix_stmts(ib.stmts, prefix)))

        # Add child memories (prefixed)
        for mem in child_flat.memories:
            memories.append(
                Memory(
                    name=f"{prefix}.{mem.name}",
                    shape=mem.shape,
                    depth=mem.depth,
                    read_ports=tuple(
                        ReadPort(
                            addr=f"{prefix}.{rp.addr}",
                            data=f"{prefix}.{rp.data}",
                            en=f"{prefix}.{rp.en}" if rp.en else None,
                            domain=f"{prefix}.{rp.domain}" if rp.domain else None,
                            transparent_for=rp.transparent_for,
                        )
                        for rp in mem.read_ports
                    ),
                    write_ports=tuple(
                        WritePort(
                            addr=f"{prefix}.{wp.addr}",
                            data=f"{prefix}.{wp.data}",
                            en=f"{prefix}.{wp.en}",
                            domain=f"{prefix}.{wp.domain}",
                            granularity=wp.granularity,
                        )
                        for wp in mem.write_ports
                    ),
                    init=mem.init,
                )
            )

        # Create port binding assignments
        binding_stmts: list[Assign] = []
        for pb in inst.bindings:
            child_port_name = f"{prefix}.{pb.port_name}"
            direction = child_port_dir.get(pb.port_name, PortDirection.INPUT)

            if direction == PortDirection.INPUT:
                # Parent drives child: parent_expr → child.port
                binding_stmts.append(
                    Assign(
                        target=child_port_name,
                        value=pb.expr,
                    )
                )
            elif direction == PortDirection.OUTPUT:
                # Child drives parent: child.port → parent_expr
                # pb.expr should be a SignalRef to the parent signal
                if isinstance(pb.expr, SignalRef):
                    binding_stmts.append(
                        Assign(
                            target=pb.expr.name,
                            value=SignalRef(shape=pb.expr.shape, name=child_port_name),
                        )
                    )
            else:
                # INOUT — bidirectional, create both directions
                binding_stmts.append(
                    Assign(
                        target=child_port_name,
                        value=pb.expr,
                    )
                )
                if isinstance(pb.expr, SignalRef):
                    binding_stmts.append(
                        Assign(
                            target=pb.expr.name,
                            value=SignalRef(shape=pb.expr.shape, name=child_port_name),
                        )
                    )

        if binding_stmts:
            comb_blocks.append(CombBlock(stmts=tuple(binding_stmts)))

    return Module(
        name=module.name,
        ports=tuple(ports),
        signals=tuple(signals),
        clock_domains=tuple(clock_domains),
        comb_blocks=tuple(comb_blocks),
        seq_blocks=tuple(seq_blocks),
        init_blocks=tuple(init_blocks),
        instances=(),
        memories=tuple(memories),
        submodules=(),
    )
