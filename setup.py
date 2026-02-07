import re
from pathlib import Path

from setuptools import setup

def get_version():
    init_file = Path(__file__).parent / "src" / "zoo" / "__init__.py"
    with open(init_file, encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find version string.")

def get_description():
    init_file = Path(__file__).parent / "src" / "zoo" / "__init__.py"
    with open(init_file, encoding="utf-8") as f:
        content = f.read()
    match = re.search(r'^__description__\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
    if match:
        return match.group(1)
    raise RuntimeError("Unable to find description string.")

setup(version=get_version(), description=get_description())
