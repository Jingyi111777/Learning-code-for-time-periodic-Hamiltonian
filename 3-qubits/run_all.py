"""Three-qubit/four-stage learning with fresh stage-to-stage result passing."""
import argparse
import dataclasses
import json
import math
from pathlib import Path
import numpy as np
from three_qubit_model import FREQUENCY_CASES, ThreeQubitOracle, EstimationPriors
from structure_learning import DEFAULT_BELL_SHOTS_PER_WINDOW, learn_structure
from bc_estimation import learn_B_and_C_for_all_targets
from omega_adaptive import learn_omega_for_all_targets, print_results as print_frequency
from bc_refinement import refine_all_targets, Config, print_results as print_refinement


def serialize(x):
    if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def run(shots=None,T_B=128.,t_short=.2,verbose=True,*,frequency_case='detuned',
        bell_shots_per_window=DEFAULT_BELL_SHOTS_PER_WINDOW):
    source = ThreeQubitOracle.from_frequency_case(frequency_case)
    prior = EstimationPriors()
    structure=learn_structure(source=source, shots_per_window=bell_shots_per_window)
    support=structure['support']
    if not support:raise RuntimeError('No nonidentity support was detected.')
    coefficients=learn_B_and_C_for_all_targets(support,source,prior,T_B=T_B,
                                               t_short=t_short,shots=shots)
    frequency=learn_omega_for_all_targets(coefficients,source=source,verbose=verbose)
    refinement_config = Config(shots=shots)
    refined=refine_all_targets(frequency,refinement_config,source=source)
    # Record simulator truth after estimation, for comparison with the recovered values.
    diagnostics = {}
    for term in source.terms:
        coefficient = coefficients.get(term.label)
        if coefficient is None:
            continue
        t_max = coefficient['max_interrogation_time']
        sine = math.sin(term.omega * t_max)
        diagnostics[term.label] = dict(
            longest_time=t_max, omega_times_longest_time=term.omega * t_max,
            sin_omega_times_longest_time=sine,
            oscillatory_contribution_to_B=term.C * sine / (term.omega * t_max),
            B_error=coefficient['B_hat'] - term.B,
            B_error_bound=coefficient['epsilon_B'],
        )
    structure_failure = structure['failure_probability']
    union_bound = None
    if structure_failure is not None:
        failures = (structure_failure
                    + sum(r['failure_probability'] for r in coefficients.values())
                    + sum(r['search_failure_probability_bound'] for r in frequency.values())
                    + (0. if shots is None else refinement_config.measurement_failure_probability))
        union_bound = min(1., failures)
    result=dict(
        model=dict(frequency_case=frequency_case, terms=source.terms, preliminary_time_cap=T_B,
                   short_time=t_short, ramsey_shots_per_quadrature=shots),
        preliminary_B_diagnostics=diagnostics,
        structure=structure,coefficients=coefficients,frequency=frequency,refinement=refined,
        union_failure_probability_bound=union_bound,
        assumptions='valid external priors, repeatable drive phase, ideal reshaping/attenuation/coherent ESU; floating-point QSP certificate',
        backend='8D three-qubit Ramsey; 64-outcome equivalent six-qubit Bell sampling; logical ESU/QSP query simulation',
        note='shots=None is infinite-shot expectation, not a finite experimental resource claim')
    if verbose:
        print('\nFrequency case:', frequency_case, FREQUENCY_CASES[frequency_case])
        print('\nLearned support:',support)
        print('Bell shots per window:', structure['shots_per_window'])
        print('Bell windows:', structure['windows'])
        print('Total Bell shots:', structure['total_bell_samples'])
        print('Theoretical Bell budget met:', structure['theorem_sample_budget_met'])
        print('B preliminary Ramsey times:',next(iter(coefficients.values()))['ramsey_times'])
        for lab,c in coefficients.items():
            print(lab,'B =',c['B_hat'],'+/-',c['epsilon_B'],
                  'C =',c['C_hat'],'+/-',c['epsilon_C'],'reshaping =',c['reshaping_group'])
        print_frequency(frequency);print_refinement(refined)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shots',type=int,default=None,help='Per Ramsey quadrature; omitted means infinite shots.')
    bell_budget = parser.add_mutually_exclusive_group()
    bell_budget.add_argument('--bell-shots-per-window', type=int, default=DEFAULT_BELL_SHOTS_PER_WINDOW,
                             help='Bell samples per window (default: %(default)s).')
    bell_budget.add_argument('--theoretical-bell-budget', action='store_const', const=None,
                             dest='bell_shots_per_window', help='Use the theoretically sufficient Bell budget.')
    parser.add_argument('--T-B',type=float,default=128.)
    parser.add_argument('--t-short',type=float,default=.2)
    parser.add_argument('--frequency-case', choices=FREQUENCY_CASES, default='detuned')
    parser.add_argument('--output', help='Output path. Defaults to results_<frequency-case>.json.')
    args=parser.parse_args()
    result=run(args.shots,args.T_B,args.t_short,frequency_case=args.frequency_case,
               bell_shots_per_window=args.bell_shots_per_window)
    output = args.output or f'results_{args.frequency_case}.json'
    Path(output).write_text(json.dumps(result,default=serialize,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Wrote',output)
