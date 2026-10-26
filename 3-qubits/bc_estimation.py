"""Richardson S and geometrically doubled Ramsey B, for discovered Pauli terms."""
from estimation_core import learn_one_target
from three_qubit_model import ThreeQubitOracle, EstimationPriors, reshaping_group


def learn_B_and_C_for_target(target_label,support,source=None,priors=None,
                             T_B=128.,t_short=.2,shots=None,failure_probability=.02,
                             quadrature_bias=0.,seed=123):
    source=ThreeQubitOracle() if source is None else source
    priors=EstimationPriors() if priors is None else priors
    group,generators=reshaping_group(target_label,tuple(support))
    result=learn_one_target(oracle=source,priors=priors,T_B=T_B,t_short=t_short,
        shots=shots,failure_probability=failure_probability,quadrature_bias=quadrature_bias,
        seed=seed,target_label=target_label)
    result.update(target_label=target_label,reshaping_group=group,
                  reshaping_generators=generators,ramsey_times=[h['t'] for h in result['history']],
                  estimation_method='Richardson S plus doubled-time Ramsey B')
    result.pop('B_hat_fit',None)
    return result


def learn_B_and_C_for_all_targets(support=None,source=None,priors=None,
                                  failure_probability=.02,seed=123,**options):
    source=ThreeQubitOracle() if source is None else source
    if support is None:
        from structure_learning import learn_structure
        support=learn_structure(source)['support']
    support=tuple(support)
    if not support or len(set(support))!=len(support):raise ValueError('Invalid learned support.')
    return {label:learn_B_and_C_for_target(label,support,source,priors,
                failure_probability=failure_probability/len(support),seed=seed+1000*i,**options)
            for i,label in enumerate(support)}

if __name__=='__main__':
    results=learn_B_and_C_for_all_targets()
    for label,r in results.items():
        print(label,'B =',r['B_hat'],'+/-',r['epsilon_B'],'C =',r['C_hat'],'+/-',r['epsilon_C'])
        print('Ramsey times:',r['ramsey_times'],'reshaping group:',r['reshaping_group'])
