"""Check the paper's plotted values against the packaged numerical records."""
import csv
import json
from pathlib import Path
import unittest

import numpy as np

from plotting.single_qubit import coefficient_data
from plotting.three_qubit import frequency_data, relative_errors


ROOT = Path(__file__).resolve().parent
THREE = ROOT / '3-qubits'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


class PaperData(unittest.TestCase):
    def test_single_qubit_uses_unrounded_estimates(self):
        data = read_json(ROOT / '1- qubit' / 'results.json')
        truth, initial, final = coefficient_data(data)
        np.testing.assert_allclose(truth, [.8, .3, .1])
        self.assertEqual(final[0], initial[0])
        np.testing.assert_allclose(100 * (initial[1:] - truth[1:]) / truth[1:],
                                   [.12871202868038, -.463417824001913], rtol=.01)
        np.testing.assert_allclose(100 * (final[1:] - truth[1:]) / truth[1:],
                                   [.00803, .0181], rtol=.01)

    def test_three_qubit_frequency_errors_and_bounds(self):
        data = read_json(THREE / 'results_detuned.json')
        truth, estimate, errors, bounds = frequency_data(data)
        records = [data['frequency'][term['label']] for term in data['model']['terms']]
        np.testing.assert_array_equal(errors, np.abs(estimate - truth))
        np.testing.assert_allclose(errors, [record['absolute_error'] for record in records])
        np.testing.assert_allclose(
            bounds,
            [record['design']['r_out'] / record['design']['T_total'] for record in records],
        )

    def test_three_qubit_relative_coefficient_errors(self):
        for name in ('results_detuned.json', 'results_detuned_1e9.json', 'results_detuned_1e10.json'):
            data = read_json(THREE / name)
            for coefficient in ('B', 'C'):
                for phase in ('before', 'after'):
                    expected = [
                        100 * data['refinement'][term['label']]['diagnostics'][f'{coefficient}_error_{phase}']
                        / abs(term[coefficient])
                        for term in data['model']['terms']
                    ]
                    np.testing.assert_allclose(relative_errors(coefficient, data, phase), expected)

    def test_attenuation_relative_errors_from_successful_trials(self):
        data = THREE / 'lambda_three_results_detuned'
        with (data / 'trials.csv').open(encoding='utf-8-sig', newline='') as stream:
            trials = list(csv.DictReader(stream))
        with (data / 'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
            summaries = list(csv.DictReader(stream))
        for summary in summaries:
            matches = [row for row in trials if float(row['lambda']) == float(summary['lambda'])]
            successful = [row for row in matches if row['success'] == 'True']
            self.assertEqual(len(successful), int(summary['successful_trials']))
            self.assertEqual(len(matches) - len(successful), int(summary['failed_trials']))
            for coefficient in ('B', 'C'):
                values = [
                    100 * abs(float(row[f'{coefficient}_hat_ref']) - float(row[f'{coefficient}_true']))
                    / abs(float(row[f'{coefficient}_true']))
                    for row in successful
                ]
                self.assertAlmostEqual(float(summary[f'{coefficient}_relative_error_mean']),
                                       float(np.mean(values)), places=13)
