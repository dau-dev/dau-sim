from __future__ import annotations

from dataclasses import dataclass, field

from dau_sim.ir.expr import Expr
from dau_sim.ir.stmt import Stmt
from dau_sim.ir.types import EdgePolarity, NetKind, PortDirection, ResetStyle, Shape


@dataclass(frozen=True)
class Signal:
    """A named net within a module."""

    name: str
    shape: Shape
    init: int = 0  # initial/reset value
    net_kind: NetKind = NetKind.WIRE  # resolution semantics for multi-driver


@dataclass(frozen=True)
class Port:
    """A module port — a signal with direction."""

    signal: Signal
    direction: PortDirection

    @property
    def name(self) -> str:
        return self.signal.name

    @property
    def shape(self) -> Shape:
        return self.signal.shape


@dataclass(frozen=True)
class ClockDomain:
    """A clock domain with clock signal, edge, and optional reset."""

    name: str
    clk: str  # signal name of the clock
    edge: EdgePolarity = EdgePolarity.POSEDGE
    rst: str | None = None  # signal name of reset, or None if resetless
    rst_style: ResetStyle = ResetStyle.SYNC
    rst_active_high: bool = True


@dataclass(frozen=True)
class CombBlock:
    """Combinational logic block (always_comb / assign).

    All statements execute whenever any referenced input signal changes.
    """

    stmts: tuple[Stmt, ...]


@dataclass(frozen=True)
class SeqBlock:
    """Sequential logic block (always_ff).

    Statements execute on the clock edge of the specified domain.
    """

    domain: str  # clock domain name
    stmts: tuple[Stmt, ...]


@dataclass(frozen=True)
class InitBlock:
    """Initial block (non-synthesizable).

    Runs once at simulation start.
    """

    stmts: tuple[Stmt, ...]


@dataclass(frozen=True)
class PortBinding:
    """Binds an instance port to a signal expression in the parent module."""

    port_name: str
    expr: Expr


@dataclass(frozen=True)
class Instance:
    """Hierarchical instantiation of another module."""

    name: str  # instance name
    module_name: str  # name of the module being instantiated
    bindings: tuple[PortBinding, ...]
    parameters: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ReadPort:
    """Memory read port descriptor."""

    addr: str  # signal name for address
    data: str  # signal name for read data
    domain: str | None = None  # None = combinational read, str = synchronous read


@dataclass(frozen=True)
class WritePort:
    """Memory write port descriptor."""

    addr: str  # signal name for address
    data: str  # signal name for write data
    en: str  # signal name for write enable
    domain: str  # clock domain for writes


@dataclass(frozen=True)
class Memory:
    """Memory array with read and write ports."""

    name: str
    shape: Shape  # shape of each element
    depth: int
    read_ports: tuple[ReadPort, ...]
    write_ports: tuple[WritePort, ...]
    init: tuple[int, ...] = ()  # initial contents


@dataclass(frozen=True)
class Module:
    """Top-level module definition.

    A named container with ports, internal signals, clock domains,
    logic blocks, submodule instances, and memories.
    """

    name: str
    ports: tuple[Port, ...] = ()
    signals: tuple[Signal, ...] = ()
    clock_domains: tuple[ClockDomain, ...] = ()
    comb_blocks: tuple[CombBlock, ...] = ()
    seq_blocks: tuple[SeqBlock, ...] = ()
    init_blocks: tuple[InitBlock, ...] = ()
    instances: tuple[Instance, ...] = ()
    memories: tuple[Memory, ...] = ()

    def port_by_name(self, name: str) -> Port | None:
        for p in self.ports:
            return p if p.name == name else None
        return None

    def signal_by_name(self, name: str) -> Signal | None:
        """Look up a signal by name, checking ports then internal signals."""
        for p in self.ports:
            if p.signal.name == name:
                return p.signal
        for s in self.signals:
            if s.name == name:
                return s
        return None

    @property
    def all_signal_names(self) -> set[str]:
        names: set[str] = set()
        for p in self.ports:
            names.add(p.signal.name)
        for s in self.signals:
            names.add(s.name)
        return names
