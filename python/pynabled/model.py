"""Robot model bindings (URDF, fixtures, chain extraction)."""

from __future__ import annotations

from pynabled._pynabled import Planar2rFixture as Planar2rFixture
from pynabled._pynabled import RobotModel as RobotModel
from pynabled._pynabled import SixDofDhFixture as SixDofDhFixture
from pynabled._pynabled import from_urdf_file_py as from_urdf_file
from pynabled._pynabled import from_urdf_str_py as from_urdf_str
from pynabled._pynabled import load_planar2r_fixture as _load_planar2r_fixture
from pynabled._pynabled import load_six_dof_dh_fixture as _load_six_dof_dh_fixture
from pynabled._pynabled import to_chain_spec_py as to_chain_spec


def load_planar2r_fixture(path: str) -> Planar2rFixture:
    return _load_planar2r_fixture(path)


def load_six_dof_dh_fixture(path: str) -> SixDofDhFixture:
    return _load_six_dof_dh_fixture(path)
