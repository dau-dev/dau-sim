from ccflow import BaseModel
from pydantic import ConfigDict

from dau_sim.ir.expr import Expr
from dau_sim.ir.stmt import Stmt
from dau_sim.ir.types import EdgePolarity, NetKind, PortDirection, ResetStyle, Shape


class Signal(BaseModel):
    """A named net within a module."""

    model_config = ConfigDict(frozen=True)

    name: str
    shape: Shape
    init: int = 0  # initial/reset value
    net_kind: NetKind = NetKind.WIRE  # resolution semantics for multi-driver


class Port(BaseModel):
    """A module port — a signal with direction."""

    model_config = ConfigDict(frozen=True)

    signal: Signal
    direction: PortDirection

    @property
    def name(self) -> str:
        return self.signal.name

    @property
    def shape(self) -> Shape:
        return self.signal.shape


class ClockDomain(BaseModel):
    """A clock domain with clock signal, edge, and optional reset."""

    model_config = ConfigDict(frozen=True)

    name: str
    clk: str  # signal name of the clock
    edge: EdgePolarity = EdgePolarity.POSEDGE
    rst: str | None = None  # signal name of reset, or None if resetless
    rst_style: ResetStyle = ResetStyle.SYNC
    rst_active_high: bool = True


class CombBlock(BaseModel):
    """Combinational logic block (always_comb / assign).

    All statements execute whenever any referenced input signal changes.
    """

    model_config = ConfigDict(frozen=True)

    stmts: tuple[Stmt, ...]


class SeqBlock(BaseModel):
    """Sequential logic block (always_ff).

    Statements execute on the clock edge of the specified domain.
    """

    model_config = ConfigDict(frozen=True)

    domain: str  # clock domain name
    stmts: tuple[Stmt, ...]


class InitBlock(BaseModel):
    """Initial block (non-synthesizable).

    Runs once at simulation start.
    """

    model_config = ConfigDict(frozen=True)

    stmts: tuple[Stmt, ...]


class PortBinding(BaseModel):
    """Binds an instance port to a signal expression in the parent module."""

    model_config = ConfigDict(frozen=True)

    port_name: str
    expr: Expr


class Instance(BaseModel):
    """Hierarchical instantiation of another module."""

    model_config = ConfigDict(frozen=True)

    name: str  # instance name
    module_name: str  # name of the module being instantiated
    bindings: tuple[PortBinding, ...]
    parameters: tuple[tuple[str, int], ...] = ()


class ReadPort(BaseModel):
    """Memory read port descriptor."""

    model_config = ConfigDict(frozen=True)

    addr: str  # signal name for address
    data: str  # signal name for read data
    en: str | None = None  # signal name for read enable (None = always enabled)
    domain: str | None = None  # None = combinational read, str = synchronous read
    transparent_for: tuple[int, ...] = ()  # write port indices for transparency


class WritePort(BaseModel):
    """Memory write port descriptor."""

    model_config = ConfigDict(frozen=True)

    addr: str  # signal name for address
    data: str  # signal name for write data
    en: str  # signal name for write enable
    domain: str  # clock domain for writes
    granularity: int = 0  # 0 = full-width write, >0 = per-granule write enable


class Memory(BaseModel):
    """Memory array with read and write ports."""

    model_config = ConfigDict(frozen=True)

    name: str
    shape: Shape  # shape of each element
    depth: int
    read_ports: tuple[ReadPort, ...]
    write_ports: tuple[WritePort, ...]
    init: tuple[int, ...] = ()  # initial contents


class Module(BaseModel):
    """Top-level module definition.

    A named container with ports, internal signals, clock domains,
    logic blocks, submodule instances, and memories.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    ports: tuple[Port, ...] = ()
    signals: tuple[Signal, ...] = ()
    clock_domains: tuple[ClockDomain, ...] = ()
    comb_blocks: tuple[CombBlock, ...] = ()
    seq_blocks: tuple[SeqBlock, ...] = ()
    init_blocks: tuple[InitBlock, ...] = ()
    instances: tuple[Instance, ...] = ()
    memories: tuple[Memory, ...] = ()
    submodules: tuple["Module", ...] = ()

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


Module.model_rebuild()
