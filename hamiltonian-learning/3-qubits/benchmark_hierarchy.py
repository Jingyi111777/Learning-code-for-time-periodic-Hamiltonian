"""Reproduce the detuned three-qubit hierarchical support-recovery benchmark."""
import argparse
import json
from pathlib import Path

from hierarchical_learning import DEFAULT_HIERARCHICAL_BELL_SHOTS, DEFAULT_THRESHOLDS
from run_all import run
from run_metadata import capture_runtime


def benchmark(trials, first_seed, shots_per_window):
    if trials < 1:
        raise ValueError('At least one trial is required.')
    records = []
    for seed in range(first_seed, first_seed + trials):
        try:
            result = run(structure_mode='hierarchical', hierarchy_seed=seed,
                         bell_shots_per_window=shots_per_window, verbose=False)
        except (ValueError, RuntimeError) as exc:
            records.append(dict(seed=seed, status='error', error=str(exc)))
            continue
        structure = result['structure']
        records.append(dict(
            seed=seed, status='completed', **structure['diagnostics'],
            additions=[row['new_terms'] for row in structure['rounds']],
            bell_samples=structure['total_bell_samples'],
            esu_queries=structure['total_esu_queries'],
        ))
    completed = [row for row in records if row['status'] == 'completed']
    return dict(
        mode='hierarchical', frequency_case='detuned', thresholds=DEFAULT_THRESHOLDS,
        ramsey_shots_per_quadrature=None,
        trials=trials, first_seed=first_seed, shots_per_window=shots_per_window,
        completed=len(completed),
        exact_support_recoveries=sum(not (row['false_positives'] or row['missed_terms'])
                                     for row in completed),
        trials_with_false_positives=sum(bool(row['false_positives']) for row in completed),
        trials_with_missed_terms=sum(bool(row['missed_terms']) for row in completed),
        records=records,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials', type=int, default=50)
    parser.add_argument('--first-seed', type=int, default=1234)
    parser.add_argument('--bell-shots-per-window', type=int, default=DEFAULT_HIERARCHICAL_BELL_SHOTS)
    parser.add_argument('--output', type=Path, default=Path('hierarchy_benchmark.json'))
    args = parser.parse_args()
    result = benchmark(args.trials, args.first_seed, args.bell_shots_per_window)
    result['run_metadata'] = capture_runtime()
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Exact support recovery: {result['exact_support_recoveries']}/{result['trials']}.")
    print(f"Wrote {args.output}")
