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
from hierarchical_learning import (
    DEFAULT_HIERARCHICAL_BELL_SHOTS, DEFAULT_THRESHOLDS, learn_hierarchically,
)
from run_metadata import capture_runtime

DEFAULT_C_MIN = .01
_MODE_BELL_BUDGET = object()


def serialize(x):
    if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def classify_coefficient(coefficient, C_min):
    """Classify from the preliminary interval and the promised modulation gap."""
    amplitude = abs(coefficient['C_hat'])
    error = coefficient['epsilon_C']
    if not np.isfinite([C_min, amplitude, error]).all() or C_min <= 0 or error < 0:
        raise ValueError('C_min must be positive and the preliminary interval must be valid.')
    if error >= C_min / 4:
        raise ValueError(f'Preliminary epsilon_C={error:.6g} must be less than C_min/4={C_min/4:.6g}. '
                         'Increase the preliminary Ramsey time or shot count to tighten the estimate.')
    if amplitude < C_min / 2:
        if amplitude <= error:
            return 'static'
    elif amplitude + error >= C_min:
        return 'modulated'
    raise ValueError('Preliminary amplitude interval is incompatible with C=0 or |C|>=C_min.')


def finish_parameter_estimation(coefficients, source, C_min, config, verbose, *, search_seed=42):
    term_types = {}
    for label, coefficient in coefficients.items():
        try:
            term_types[label] = classify_coefficient(coefficient, C_min)
        except ValueError as exc:
            raise ValueError(f'{label}: {exc}') from exc
    static = {lab: c for lab,c in coefficients.items() if term_types[lab]=='static'}
    modulated = {lab: c for lab,c in coefficients.items() if term_types[lab]=='modulated'}
    frequency = (learn_omega_for_all_targets(modulated,source=source,verbose=verbose,seed=search_seed)
                 if modulated else {})
    refined=refine_all_targets(frequency,config,source=source,
                               static_coefficients=static)
    return term_types, frequency, refined


def run(shots=None,T_B=128.,t_short=.2,verbose=True,*,frequency_case='detuned',
        bell_shots_per_window=_MODE_BELL_BUDGET,C_min=DEFAULT_C_MIN, source=None,
        structure_mode='single-pass', thresholds=DEFAULT_THRESHOLDS, hierarchy_seed=1234):
    if not np.isfinite(C_min) or C_min <= 0:
        raise ValueError('C_min must be positive and finite.')
    if structure_mode not in ('single-pass', 'hierarchical'):
        raise ValueError('Unknown structure-learning mode.')
    if source is None:
        source = ThreeQubitOracle.from_frequency_case(frequency_case)
    else:
        frequency_case = 'custom'
    prior = EstimationPriors()
    refinement_config = Config(shots=shots)
    hierarchical = structure_mode == 'hierarchical'
    seeds = (dict(hierarchy=hierarchy_seed) if hierarchical else
             dict(bell=1234, ramsey_bc=123, bbht=42, refinement=refinement_config.seed))
    if bell_shots_per_window is _MODE_BELL_BUDGET:
        bell_shots_per_window = (DEFAULT_HIERARCHICAL_BELL_SHOTS if hierarchical
                                else DEFAULT_BELL_SHOTS_PER_WINDOW)
    if hierarchical:
        def finish(coefficients, search_seed, refinement_seed):
            config = dataclasses.replace(refinement_config, seed=refinement_seed)
            return finish_parameter_estimation(
                coefficients, source, C_min, config, verbose, search_seed=search_seed,
            )

        structure, coefficients, term_types, frequency, refined = learn_hierarchically(
            source, prior, finish, thresholds=thresholds, bell_shots=bell_shots_per_window,
            seed=hierarchy_seed, shots=shots, T_B=T_B, t_short=t_short,
        )
    else:
        structure = learn_structure(source=source, shots_per_window=bell_shots_per_window,
                                    seed=seeds['bell'])
        if not structure['support']:
            raise RuntimeError('No nonidentity support was detected.')
        coefficients = learn_B_and_C_for_all_targets(
            structure['support'], source, prior, T_B=T_B, t_short=t_short, shots=shots,
            seed=seeds['ramsey_bc'],
        )
        term_types, frequency, refined = finish_parameter_estimation(
            coefficients, source, C_min, refinement_config, verbose,
            search_seed=seeds['bbht'],
        )
    support = structure['support']
    # Record simulator truth after estimation, for comparison with the recovered values.
    diagnostics = {}
    for term in source.terms:
        coefficient = coefficients.get(term.label)
        if coefficient is None:
            continue
        t_max = coefficient['max_interrogation_time']
        angle = term.omega * t_max if term.C != 0 else None
        sine = math.sin(angle) if angle is not None else None
        diagnostics[term.label] = dict(
            longest_time=t_max, omega_times_longest_time=angle,
            sin_omega_times_longest_time=sine,
            oscillatory_contribution_to_B=term.C * sine / angle if angle is not None else 0.,
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
    static_frequency = {
        lab: dict(target_label=lab, status='not_applicable', omega_hat=None,
                  frequency_error_bound=None, esu_queries=0, search_failure_probability_bound=0.)
        for lab in support if term_types[lab] == 'static'
    }
    result=dict(
        model=dict(frequency_case=frequency_case, terms=source.terms, preliminary_time_cap=T_B,
                   short_time=t_short, ramsey_shots_per_quadrature=shots, C_min=C_min),
        preliminary_B_diagnostics=diagnostics,
        structure=structure,coefficients=coefficients,term_types=term_types,
        frequency={**frequency, **static_frequency},refinement=refined,
        union_failure_probability_bound=union_bound,
        assumptions='valid external priors including C=0 or |C|>=C_min; repeatable drive phase; ideal reshaping/attenuation/coherent ESU; floating-point QSP certificate',
        backend='8D three-qubit Ramsey; 64-outcome equivalent six-qubit Bell sampling; logical ESU/QSP query simulation',
        note='shots=None is infinite-shot expectation, not a finite experimental resource claim')
    result['run_metadata'] = dict(**capture_runtime(), structure_mode=structure_mode, seeds=seeds,
                                  ramsey_shots_per_quadrature=shots)
    if verbose:
        print('\nFrequency case:', frequency_case, FREQUENCY_CASES.get(frequency_case))
        print('\nLearned support:', support)
        if hierarchical:
            for stage in structure['rounds']:
                print(f"Round {stage['round']}, a_j={stage['a_j']}: "
                      f"confirmed={stage['confirmed']}, added={stage['new_terms']}, "
                      f"compensated={[term.label for term in stage['compensation_in']]}, "
                      f"compensation error bound={stage['compensation_error_bound']:.6g}, "
                      f"Bell samples={stage['proposal_bell_samples']}+{stage['validation_bell_samples']}, "
                      f"ESU queries={stage['esu_queries']}")
            print('Total ESU queries across rounds:', structure['total_esu_queries'])
        else:
            print('Bell shots per window:', structure['shots_per_window'])
            print('Bell windows:', structure['windows'])
        print('Total Bell shots:', structure['total_bell_samples'])
        if not hierarchical:
            print('Theoretical Bell budget met:', structure['theorem_sample_budget_met'])
        if coefficients:
            print('B preliminary Ramsey times:', next(iter(coefficients.values()))['ramsey_times'])
        for lab,c in coefficients.items():
            print(lab,'B =',c['B_hat'],'+/-',c['epsilon_B'],
                  'C =',c['C_hat'],'+/-',c['epsilon_C'],'reshaping =',c['reshaping_group'])
        if frequency:
            print_frequency(frequency)
        if static_frequency:
            print('Static terms (frequency search skipped):', ', '.join(static_frequency))
        print_refinement(refined)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shots',type=int,default=None,help='Per Ramsey quadrature; omitted means infinite shots.')
    bell_budget = parser.add_mutually_exclusive_group()
    bell_budget.add_argument('--bell-shots-per-window', type=int, default=_MODE_BELL_BUDGET,
                             help=(f'Bell samples per window. Defaults: {DEFAULT_BELL_SHOTS_PER_WINDOW:.0e}'
                                   f' single-pass, {DEFAULT_HIERARCHICAL_BELL_SHOTS:.0e} hierarchical.'))
    bell_budget.add_argument('--theoretical-bell-budget', action='store_const', const=None,
                             dest='bell_shots_per_window',
                             help='Use the single-pass reference Bell budget for each sampling pass.')
    parser.add_argument('--T-B',type=float,default=128.)
    parser.add_argument('--t-short',type=float,default=.2)
    parser.add_argument('--frequency-case', choices=FREQUENCY_CASES, default='detuned')
    parser.add_argument('--c-min',type=float,default=DEFAULT_C_MIN,
                        help='Promised lower bound on nonzero modulation amplitudes.')
    parser.add_argument('--structure-mode', choices=('single-pass', 'hierarchical'), default='single-pass')
    parser.add_argument('--amplitude-thresholds', type=float, nargs='+', default=DEFAULT_THRESHOLDS,
                        help='Decreasing thresholds for hierarchical mode. The final value is a_min.')
    parser.add_argument('--hierarchy-seed', type=int, default=1234)
    parser.add_argument('--output', help='Output path. Hierarchical mode uses a separate default file.')
    args=parser.parse_args()
    result=run(args.shots,args.T_B,args.t_short,frequency_case=args.frequency_case,
               bell_shots_per_window=args.bell_shots_per_window,C_min=args.c_min,
               structure_mode=args.structure_mode, thresholds=args.amplitude_thresholds,
               hierarchy_seed=args.hierarchy_seed)
    suffix = '_hierarchical' if args.structure_mode == 'hierarchical' else ''
    output = args.output or f'results_{args.frequency_case}{suffix}.json'
    Path(output).write_text(json.dumps(result,default=serialize,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Wrote',output)
