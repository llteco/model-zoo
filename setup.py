"""Fallback build script for toolchains that cannot read the
``[tool.setuptools]`` / ``[project]`` tables from pyproject.toml
(e.g. old pip or ``setup.py``-only flows on Python 3.10 targets).

pyproject.toml stays authoritative; this file only mirrors the src-layout
package discovery and fills the dynamic version/description fields.
"""

import re
from pathlib import Path

from setuptools import find_packages, setup

_src = Path(__file__).parent / "src" / "zoo" / "__init__.py"
_version = re.search(r'__version__ = "([^"]+)"', _src.read_text(encoding="utf-8"))
_description = re.search(r'__description__ = "([^"]+)"', _src.read_text(encoding="utf-8"))

setup(
    version=_version.group(1) if _version else "0.0.0",
    description=_description.group(1) if _description else "",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
