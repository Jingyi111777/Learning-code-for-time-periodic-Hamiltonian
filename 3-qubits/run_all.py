"""Three-qubit/four-stage learning with fresh stage-to-stage result passing."""
import argparse
import dataclasses
import json
from pathlib import Path
import numpy as np
from three_qubit_model import ThreeQubitOracle, EstimationPriors
from structure_learning import learn_structure
from bc_estimation import learn_B_and_C_for_all_targets
from omega_adaptive import learn_omega_for_all_targets, print_results as print_frequency
from bc_refinement import refine_all_targets, Config, print_results as print_refinement


def serialize(x):
    if dataclasses.is_dataclass(x):return dataclasses.asdict(x)
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    raise TypeError(type(x).__name__)


def run(shots=None,T_B=128.,t_short=.2,verbose=True):
    source=ThreeQubitOracle();prior=EstimationPriors()
    structure=learn_structure(source=source)
    support=structure['support']
    if not support:raise RuntimeError('No nonidentity support was detected.')
    coefficients=learn_B_and_C_for_all_targets(support,source,prior,T_B=T_B,
                                               t_short=t_short,shots=shots)
    frequency=learn_omega_for_all_targets(coefficients,source=source,verbose=verbose)
    refined=refine_all_targets(frequency,Config(shots=shots),source=source)
    failures=(structure['failure_probability']+sum(r['failure_probability'] for r in coefficients.values())
              +sum(r['search_failure_probability_bound'] for r in frequency.values())
              +(0. if shots is None else .01))
    result=dict(structure=structure,coefficients=coefficients,frequency=frequency,refinement=refined,
        union_failure_probability_bound=min(1.,failures),
        assumptions='valid external priors, repeatable drive phase, ideal reshaping/attenuation/coherent ESU; floating-point QSP certificate',
        backend='8D three-qubit Ramsey; 64-outcome equivalent six-qubit Bell sampling; logical ESU/QSP query simulation',
        note='shots=None is infinite-shot expectation, not a finite experimental resource claim')
    if verbose:
        print('\nLearned support:',support)
        print('B preliminary Ramsey times:',next(iter(coefficients.values()))['ramsey_times'])
        for lab,c in coefficients.items():
            print(lab,'B =',c['B_hat'],'+/-',c['epsilon_B'],
                  'C =',c['C_hat'],'+/-',c['epsilon_C'],'reshaping =',c['reshaping_group'])
        print_frequency(frequency);print_refinement(refined)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--shots',type=int,default=None)
    parser.add_argument('--T-B',type=float,default=128.)
    parser.add_argument('--t-short',type=float,default=.2)
    parser.add_argument('--output',default='results.json')
    args=parser.parse_args()
    result=run(args.shots,args.T_B,args.t_short)
    Path(args.output).write_text(json.dumps(result,default=serialize,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Wrote',args.output)
