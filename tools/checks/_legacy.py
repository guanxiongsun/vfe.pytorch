"""Where the legacy mmdetection tree lives, for the harnesses that read it.

From vfe.pytorch 2.0 the mmdet/ tree is no longer part of this repository: it
is preserved at the ``v1.0.0`` tag and checked out separately as the parity
oracle (see docs/parity.md). Two harnesses still read data files out of it, so
they ask here instead of assuming a path.

Searched in order, first one containing ``mmdet/__init__.py`` wins:

1. ``$VFE_LEGACY_ROOT``
2. ``../vfe.legacy`` next to this repository -- the worktree docs/parity.md creates
3. this repository itself -- only before the 2.0 removal, and only in a v1 checkout

Kept parsable by Python 3.8: the legacy conda env runs these harnesses too.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# A moved or copied harness would otherwise resolve REPO_ROOT to some unrelated
# directory and quietly check the wrong tree.
assert (REPO_ROOT / "vfe").is_dir(), "tools/checks/ must stay two levels below the repo root"

WORKTREE = REPO_ROOT.parent / "vfe.legacy"


def legacy_root():
    """The root of a legacy (mmdet 2.19.1) checkout, as a ``Path``."""
    env = os.environ.get("VFE_LEGACY_ROOT")
    candidates = [Path(env)] if env else []
    candidates += [WORKTREE, REPO_ROOT]
    for candidate in candidates:
        if (candidate / "mmdet" / "__init__.py").is_file():
            return candidate
    looked = ", ".join(str(c) for c in candidates)
    raise SystemExit(
        f"no legacy mmdet tree found (looked in: {looked}).\n"
        f"Create one with:\n"
        f"    git worktree add {WORKTREE} v1\n"
        f"or point VFE_LEGACY_ROOT at an existing checkout."
    )


def legacy_file(rel):
    """``rel`` inside the legacy tree; raises if it is not there."""
    path = legacy_root() / rel
    if not path.exists():
        raise SystemExit(f"{rel} is missing from the legacy tree at {legacy_root()}")
    return path
