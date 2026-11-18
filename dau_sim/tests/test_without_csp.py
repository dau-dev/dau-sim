"""csp has no Linux ARM wheels, so dau-sim installs there without it. The
integrations other packages build on must import without csp."""

import subprocess
import sys

_BLOCK_CSP = """
import sys

class _NoCsp:
    def find_spec(self, name, path=None, target=None):
        if name == "csp" or name.startswith("csp."):
            raise ImportError("csp is not installed")

sys.meta_path.insert(0, _NoCsp())
import dau_sim
import dau_sim.integrations.cocotb
import dau_sim.integrations.profiles
import dau_sim.integrations.verilator
import dau_sim.integrations.verilator_profiles
assert "csp" not in sys.modules
"""


def test_the_integrations_import_without_csp() -> None:
    result = subprocess.run([sys.executable, "-c", _BLOCK_CSP], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
