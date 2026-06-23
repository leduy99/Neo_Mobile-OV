"""SANA-video third-party code integrated into Omni-Video-smolvlm2."""

import os
import sys

_pkg_dir = os.path.dirname(__file__)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)
