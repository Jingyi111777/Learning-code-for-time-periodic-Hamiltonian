"""Run the five stages with fresh results. No copy/paste of fitted numbers."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
from structure_learning import DEFAULT_BELL_SHOTS_PER_WINDOW, learn_structure
from x_coefficient_estimation import learn_x_coefficient
from bc_estimation import learn_BC
from omega_adaptive import learn_omega_for_single_qubit
from bc_refinement import refine_single_qubit, Config


def serializable(x):
    if dataclasses.is_dataclass(x): return dataclasses.asdict(x)
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def run(shots=None, verbose=True, *, bell_shots_per_window=DEFAULT_BELL_SHOTS_PER_WINDOW):
    """Run the example. A None Bell budget selects the theoretically sufficient budget."""
    structure=learn_structure(shots_per_window=bell_shots_per_window)
    # X/Z routines implement the identified single-qubit model. Fail explicitly
    # for unsupported structures rather than silently ignoring newly found terms.
    if set(structure['support'])!=set(('X','Z')):
        raise RuntimeError('Identified structure is outside this X+Z single-qubit model.')
    x=learn_x_coefficient(shots=shots)
    bc=learn_BC(shots=shots)
    frequency=learn_omega_for_single_qubit(bc,verbose=verbose)
    refinement=refine_single_qubit(frequency,Config(shots=shots))
    bounds=[structure['failure_probability'],x['failure_probability'],
            bc['failure_probability'],frequency['search_failure_probability_bound'],
            refinement['refinement_measurement_failure_probability']]
    union_bound = min(1., sum(bounds)) if all(bound is not None for bound in bounds) else None
    guarantee = ('conditional on correct priors, model, ideal ESU and floating-point QSP certificate'
                 if union_bound is not None else None)
    results=dict(structure=structure,X=x,BC=bc,frequency=frequency,refinement=refinement,
                 union_failure_probability_bound=union_bound,
                 guarantee=guarantee,
                 note='Search runtime cap can raise. None shots is the infinite-shot limit, not finite experimental cost.')
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shots',type=int,default=None,help='Per Ramsey quadrature; omitted means infinite shots.')
    bell_budget = parser.add_mutually_exclusive_group()
    bell_budget.add_argument('--bell-shots-per-window', type=int, default=DEFAULT_BELL_SHOTS_PER_WINDOW,
                             help='Bell samples per window (default: %(default)s).')
    bell_budget.add_argument('--theoretical-bell-budget', action='store_const', const=None,
                             dest='bell_shots_per_window',
                             help='Use the theoretically sufficient Bell budget.')
    parser.add_argument('--output',default='results.json')
    args=parser.parse_args()
    r=run(args.shots, bell_shots_per_window=args.bell_shots_per_window)
    Path(args.output).write_text(json.dumps(r,default=serializable,ensure_ascii=False,indent=2),encoding='utf-8')
    print('support =',r['structure']['support'])
    print('Bell shots per window =', r['structure']['shots_per_window'])
    print('Bell windows =', r['structure']['windows'])
    print('Total Bell shots =', r['structure']['total_bell_samples'])
    print('Theoretical Bell budget met =', r['structure']['theorem_sample_budget_met'])
    print('A =',r['X']['A_hat'],'+/-',r['X']['error_bound'])
    print('omega =',r['frequency']['omega_hat'],'+/-',r['frequency']['frequency_error_bound'])
    for kind in ('B','C'):
        v=r['refinement'][kind]
        print(kind,'=',v['estimate'],'+/-',v['error_bound'],'status =',v['status'])
    print('Wrote',args.output)
