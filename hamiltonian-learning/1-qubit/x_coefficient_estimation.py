"""Static X estimation by confidence intervals for Ramsey phases.
Keeps the correct signal <Z>-i<Y>=exp(2i A t), accounts for sampling uncertainty.
"""
import numpy as np
from learning_common import ModelOracle, phase_error, interval_update


def learn_x_coefficient(epsilon=.01,a_init=-1.,b_init=1.,shots=None,
                        failure_probability=.01,quadrature_bias=0.,seed=1234,
                        max_scales=30,oracle=None):
    if not np.isfinite([epsilon,a_init,b_init]).all() or epsilon<=0 or a_init>=b_init:
        raise ValueError('Invalid target/prior interval.')
    if not isinstance(max_scales,int) or max_scales<=0:
        raise ValueError('max_scales must be a positive integer.')
    oracle=ModelOracle() if oracle is None else oracle
    eps_phi=phase_error(shots,failure_probability/max_scales,quadrature_bias)
    intervals=[(a_init,b_init)]
    rng=np.random.default_rng(seed)
    # The centered phase at t0 spans [-pi/2,pi/2].
    t0=np.pi/(2*(b_init-a_init))
    history=[]
    for j in range(max_scales):
        if (intervals[-1][1]-intervals[0][0])/2<=epsilon:
            break
        t=t0*2.**j
        d=oracle.ramsey('X',t,shots,rng)
        phi=.5*np.arctan2(d['S'],d['R'])
        intervals=interval_update(intervals,phi,t,eps_phi/2)
        history.append(dict(t=t,measurement=d,intervals=intervals.copy()))
    lo,hi=intervals[0][0],intervals[-1][1]
    radius=(hi-lo)/2
    return dict(A_hat=(lo+hi)/2,error_bound=radius,final_interval=(lo,hi),
                intervals=intervals,iterations=len(history),history=history,
                target_error=epsilon,status='target_reached' if radius<=epsilon else 'max_scales',
                failure_probability=0. if shots is None else failure_probability,
                backend='ideal_isolated_or_user_measurement_callback',
                total_sensing_time=None if shots is None else float(2*shots*sum(x['t'] for x in history)))

if __name__=='__main__':
    r=learn_x_coefficient()
    print({k:v for k,v in r.items() if k!='history'})
