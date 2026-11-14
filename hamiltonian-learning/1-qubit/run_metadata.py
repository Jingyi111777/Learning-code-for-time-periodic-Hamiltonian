"""Runtime information for reproducible simulation outputs."""
import os
from pathlib import Path
import platform

import numpy as np
import scipy


def capture_runtime():
    version = Path(__file__).resolve().parents[1] / 'VERSION'
    return dict(
        code_version=version.read_text(encoding='utf-8').strip() if version.is_file() else None,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        system=platform.system(),
        machine=platform.machine(),
        thread_environment={key: os.environ.get(key)
                            for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS')},
    )
