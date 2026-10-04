"""The wheel's build hook: hands' wheel carries the fritter binary, built from fritter/ for macOS arm64."""

import os
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# Go 1.25's oldest supported macOS is 12, so that is the oldest a wheel with its binary in it installs on.
TAG = "py3-none-macosx_12_0_arm64"
# Where in the installed package fritter lands: hands/bin/fritter.
PACKAGED = "hands/bin/fritter"


class FritterHook(BuildHookInterface[Any]):
    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        # An editable install is a checkout, which runs fritter from its source as it always has: no Go asked of it.
        if version == "editable":
            return
        root = Path(self.root)
        built = root / "build" / "fritter"
        environ = {**os.environ, "GOOS": "darwin", "GOARCH": "arm64"}
        # [LAW:no-silent-failure] no Go, or a failed build, fails the wheel: a hands wheel without fritter is no release.
        subprocess.run(["go", "build", "-trimpath", "-o", str(built), "."], cwd=root / "fritter", env=environ, check=True)
        build_data["force_include"][str(built)] = PACKAGED
        build_data["pure_python"] = False
        build_data["tag"] = TAG
