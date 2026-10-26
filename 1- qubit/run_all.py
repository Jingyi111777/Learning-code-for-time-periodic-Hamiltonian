"""Run the five stages with fresh results. No copy/paste of fitted numbers."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
from structure_learning import learn_structure
from x_coefficient_estimation import learn_x_coefficient
from bc_estimation import learn_BC
from omega_adaptive import learn_omega_for_single_qubit
from bc_refinement import refine_single_qubit, Config


def serializable(x):
    if dataclasses.is_dataclass(x): return dataclasses.asdict(x)
    if isinstance(x,np.ndarray): return x.tolist()
    if isinstance(x,np.generic): return x.item()
    raise TypeError(type(x).__name__)


def run(shots=None,verbose=True):
    structure=learn_structure()
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
    results=dict(structure=structure,X=x,BC=bc,frequency=frequency,refinement=refinement,
                 union_failure_probability_bound=min(1.,sum(bounds)),
                 guarantee='conditional on correct priors, model, ideal ESU and floating-point QSP certificate',
                 note='Search runtime cap can raise. None shots is the infinite-shot limit, not finite experimental cost.')
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shots',type=int,default=None,help='Per Ramsey quadrature; omitted means infinite shots.')
    parser.add_argument('--output',default='results.json')
    args=parser.parse_args()
    r=run(args.shots)
    Path(args.output).write_text(json.dumps(r,default=serializable,ensure_ascii=False,indent=2),encoding='utf-8')
    print('support =',r['structure']['support'])
    print('A =',r['X']['A_hat'],'+/-',r['X']['error_bound'])
    print('omega =',r['frequency']['omega_hat'],'+/-',r['frequency']['frequency_error_bound'])
    for kind in ('B','C'):
        v=r['refinement'][kind]
        print(kind,'=',v['estimate'],'+/-',v['error_bound'],'status =',v['status'])
    print('Wrote',args.output)
