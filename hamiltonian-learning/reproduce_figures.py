"""Reproduce the paper's Figures 2–4 and record their inputs and hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager, rc_context
import numpy as np
from PIL import __version__ as pillow_version
import scipy

from plotting.single_qubit import plot_single_qubit
from plotting.three_qubit import plot_three_qubit


ROOT = Path(__file__).resolve().parent
INPUTS = (
    ROOT / '1- qubit' / 'results.json',
    ROOT / '3-qubits' / 'results_detuned.json',
    ROOT / '3-qubits' / 'results_detuned_1e9.json',
    ROOT / '3-qubits' / 'results_detuned_1e10.json',
)
FIGURES = (
    'single_qubit_coefficient_bars.png',
    'three_qubit_frequency_bars.png',
    'three_qubit_coefficient_bars.png',
)


def rerun_simulations(output):
    results = output / 'simulation_results'
    results.mkdir(exist_ok=True)
    jobs = (
        (INPUTS[0], 'single_qubit.json', ()),
        (INPUTS[1], 'three_qubit.json', ('--frequency-case', 'detuned')),
        (INPUTS[2], 'three_qubit_1e9.json', ('--frequency-case', 'detuned', '--shots', '1000000000')),
        (INPUTS[3], 'three_qubit_1e10.json', ('--frequency-case', 'detuned', '--shots', '10000000000')),
    )
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    paths = []
    for source, name, options in jobs:
        target = results / name
        subprocess.run([sys.executable, 'run_all.py', *options, '--output', str(target)],
                       cwd=source.parent, env=env, check=True)
        paths.append(target)
    return paths


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce(output, *, rerun=False):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = rerun_simulations(output) if rerun else INPUTS
    single, baseline, finite_9, finite_10 = [json.loads(path.read_text(encoding='utf-8'))
                                            for path in paths]
    try:
        font_manager.findfont('Liberation Serif', fallback_to_default=False)
        font = 'Liberation Serif'
    except ValueError:
        font = 'DejaVu Serif'
        print('Using DejaVu Serif. Install Liberation Serif to match the paper typography.')

    with rc_context():
        plot_single_qubit(single, output / FIGURES[0], font)
    with rc_context():
        plot_three_qubit(baseline, finite_9, finite_10, output, font)

    sources = (paths[:1], paths[1:2], paths[1:])
    manifest = dict(
        code_version=(ROOT / 'VERSION').read_text(encoding='utf-8').strip(),
        input_mode='rerun' if rerun else 'archived',
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         scipy=scipy.__version__, matplotlib=matplotlib.__version__,
                         pillow=pillow_version, font=font),
        figures={
            name: dict(sha256=sha256(output / name),
                       inputs=[dict(path=str(path.relative_to(output) if path.is_relative_to(output)
                                             else path.relative_to(ROOT)), sha256=sha256(path))
                               for path in inputs])
            for name, inputs in zip(FIGURES, sources)
        },
    )
    (output / 'figure_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote Figures 2–4 and figure_manifest.json to {output}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('figures'))
    parser.add_argument('--rerun', action='store_true', help='Rerun the four simulations before plotting.')
    args = parser.parse_args()
    reproduce(args.output_dir, rerun=args.rerun)
