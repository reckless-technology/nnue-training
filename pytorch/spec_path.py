# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Jonny Reckless
"""Put `spec/` on `sys.path`, so the modules here can `import rnnue1`.

`spec/rnnue1.py` is the one description of the net container and the architecture, and it deliberately
sits in `spec/` rather than beside the trainer: the engines consume it from there too, and a second copy
next to this trainer is exactly the duplication this repository exists to end. Nothing here is a package
-- `train.py` and `verify.py` are run as scripts -- so a relative import is not available and the path
has to be arranged before the import. Importing this module is what arranges it:

    import spec_path  # noqa: F401  (puts spec/ on sys.path for the rnnue1 import below)
    from rnnue1 import HIDDEN_SIZE

The engines do the same thing inline, each in a single tool (`zenith/tools/stamp_net.py`). Here four
modules need it, so it is written once.
"""
import os
import sys

_SPEC_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "spec")

if _SPEC_DIRECTORY not in sys.path:
    sys.path.insert(0, _SPEC_DIRECTORY)
