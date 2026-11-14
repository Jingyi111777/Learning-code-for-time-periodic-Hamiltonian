"""Run the five stages with fresh results. No copy/paste of fitted numbers."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
from learning_common import ModelOracle
from structure_learning import DEFAULT_BELL_SHOTS_PER_WINDOW, learn_structure
from x_coefficient_estimation import learn_x_coefficient
from bc_estimation import DEFAULT_C_MIN, classify_modulation, learn_BC
from omega_adaptive import learn_omega_for_single_qubit
from bc_refinement import refine_single_qubit, Config
from run_metadata import capture_runtime


def serializable(x):
    if dataclasses.is_dataclass(x): return dataclasses.asdict(x)
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def run(shots=None, verbose=True, *, bell_shots_per_window=DEFAULT_BELL_SHOTS_PER_WINDOW,
        oracle=None, C_min=DEFAULT_C_MIN):
    """Learn the X/Z model under the modulation prior C=0 or |C|>=C_min."""
    oracle = ModelOracle() if oracle is None else oracle
    config = Config(shots=shots)
    seeds = dict(bell=1234, ramsey_x=1234, ramsey_bc=123, bbht=42, refinement=config.seed)
    structure=learn_structure(oracle=oracle,shots_per_window=bell_shots_per_window,
                              seed=seeds['bell'])
    # X/Z routines implement the identified single-qubit model. Fail explicitly
    # for unsupported structures rather than silently ignoring newly found terms.
    if set(structure['support'])!=set(('X','Z')):
        raise RuntimeError('Identified structure is outside this X+Z single-qubit model.')
    x=learn_x_coefficient(shots=shots,oracle=oracle,seed=seeds['ramsey_x'])
    bc=learn_BC(shots=shots,oracle=oracle,seed=seeds['ramsey_bc'])
    term_type = classify_modulation(bc, C_min)
    if term_type == 'static':
        frequency = dict(target_label='Z', status='not_applicable', omega_hat=None,
                         frequency_error_bound=None, esu_queries=0,
                         search_failure_probability_bound=0.)
        refinement = refine_single_qubit(config=config, oracle=oracle, static_coefficient=bc)
        guarantee_condition = 'correct priors, model and ideal reshaping'
        seeds.pop('bbht')
    else:
        frequency=learn_omega_for_single_qubit(bc,verbose=verbose,oracle=oracle,seed=seeds['bbht'])
        if verbose:
            design = frequency['design']
            print(f"Frequency design (Z): r_in={design.r_in!r}, r_out={design.r_out!r}, "
                  f"T_omega={design.T_total!r}, K={design.number_of_bins}, "
                  f"qsp_degree={frequency['qsp_degree']}, epsilon_corr={design.correction_bound!r}, "
                  f"epsilon_omega={frequency['frequency_error_bound']!r}")
        refinement=refine_single_qubit(frequency,config,oracle=oracle)
        guarantee_condition = 'correct priors, model, ideal ESU and floating-point QSP certificate'
    bounds=[structure['failure_probability'],x['failure_probability'],
            bc['failure_probability'],frequency['search_failure_probability_bound'],
            refinement['refinement_measurement_failure_probability']]
    union_bound = min(1., sum(bounds)) if all(bound is not None for bound in bounds) else None
    guarantee = f'conditional on {guarantee_condition}' if union_bound is not None else None
    results=dict(structure=structure,X=x,BC=bc,frequency=frequency,refinement=refinement,
                 union_failure_probability_bound=union_bound,
                 guarantee=guarantee,
                 note='None shots is the infinite-shot limit, not finite experimental cost.')
    results['model'] = dict(A=oracle.A, B=oracle.B, C=oracle.C, omega=oracle.omega)
    results['term_type'] = term_type
    assumptions = ('ideal reshaping; finite Ramsey shots add sampling noise' if term_type == 'static'
                   else 'ideal reshaping and coherent ESU; finite Ramsey shots add sampling noise')
    results['run_metadata'] = dict(**capture_runtime(), seeds=seeds,
                                   C_min=C_min,
                                   ramsey_shots_per_quadrature=shots,
                                   assumptions=assumptions)
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shots',type=int,default=None,help='Per Ramsey quadrature; omitted means infinite shots.')
    parser.add_argument('--c', type=float, default=ModelOracle.C, help='Simulated modulation amplitude C.')
    parser.add_argument('--c-min', type=float, default=DEFAULT_C_MIN,
                        help='Prior: a nonzero |C| is at least this value.')
    bell_budget = parser.add_mutually_exclusive_group()
    bell_budget.add_argument('--bell-shots-per-window', type=int, default=DEFAULT_BELL_SHOTS_PER_WINDOW,
                             help='Bell samples per window (default: %(default)s).')
    bell_budget.add_argument('--theoretical-bell-budget', action='store_const', const=None,
                             dest='bell_shots_per_window',
                             help='Use the theoretically sufficient Bell budget.')
    parser.add_argument('--output',default='results.json')
    args=parser.parse_args()
    oracle = ModelOracle(C=args.c, omega=None if args.c == 0 else ModelOracle.omega)
    r=run(args.shots, bell_shots_per_window=args.bell_shots_per_window, oracle=oracle, C_min=args.c_min)
    Path(args.output).write_text(json.dumps(r,default=serializable,ensure_ascii=False,indent=2),encoding='utf-8')
    print('support =',r['structure']['support'])
    print('Bell shots per window =', r['structure']['shots_per_window'])
    print('Bell windows =', r['structure']['windows'])
    print('Total Bell shots =', r['structure']['total_bell_samples'])
    print('Theoretical Bell budget met =', r['structure']['theorem_sample_budget_met'])
    print('A =',r['X']['A_hat'],'+/-',r['X']['error_bound'])
    if r['term_type'] == 'static':
        print('Static Z: frequency search skipped.')
    else:
        print('omega =',r['frequency']['omega_hat'],'+/-',r['frequency']['frequency_error_bound'])
    for kind in ('B','C'):
        v=r['refinement'][kind]
        print(kind,'=',v['estimate'],'+/-',v['error_bound'],'status =',v['status'])
    print('Wrote',args.output)
