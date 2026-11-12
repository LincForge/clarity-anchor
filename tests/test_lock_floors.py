"""Pin security floors for locked dependencies.

Dependabot advisories against Pillow < 13.0.0 are resolved by the lock. This
test fails if a future re-lock drops Pillow back below the patched version.
"""

import re
import tomllib
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "uv.lock"

FLOORS = {
    "pillow": (13, 0, 0),
}


def _version_tuple(version: str) -> tuple[int, ...]:
    parts = []
    for piece in version.split("."):
        match = re.match(r"\d+", piece)
        if match is None:
            break
        parts.append(int(match.group()))
        if match.group() != piece:
            break
    return tuple(parts)


def _locked_versions() -> dict[str, str]:
    data = tomllib.loads(LOCK.read_text())
    return {pkg["name"].lower(): pkg["version"] for pkg in data["package"]}


def test_locked_packages_meet_security_floors():
    locked = _locked_versions()
    for name, floor in FLOORS.items():
        assert name in locked, f"{name} missing from uv.lock"
        assert _version_tuple(locked[name]) >= floor, (
            f"{name} {locked[name]} is below the patched floor {floor}"
        )


def test_version_tuple_parses_prerelease_suffix():
    assert _version_tuple("13.0.0rc1") == (13, 0, 0)
    assert _version_tuple("12.3.0") < (13, 0, 0)
