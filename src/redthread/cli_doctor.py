"""Compatibility shim for legacy CLI doctor imports."""

import shutil
from urllib.request import urlopen

from redthread.cli.doctor import *  # noqa: F403

__all__ = [  # noqa: F405
    "DoctorCheck",
    "STATUS_LABEL",
    "STATUS_STYLE",
    "collect_doctor_checks",
    "render_doctor_report",
    "run_doctor",
    "shutil",
    "urlopen",
]
