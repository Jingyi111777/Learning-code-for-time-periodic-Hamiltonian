
import math
import numpy as np
from learning_common import Priors, ModelOracle, phase_error


def learn_BC(oracle=None, priors=None, T_B=32., t_short=.08, shots=None,
             failure_probability=.02, quadrature_bias=0., seed=123):
    oracle=ModelOracle() if oracle is None else oracle
    p=Priors() if priors is None else priors
    p.validate()
    if not np.isfinite([T_B,t_short,failure_probability]).all() or min(T_B,t_short)<=0:
        raise ValueError('Times must be positive and finite.')
    if not 0<failure_probability<1:
        raise ValueError('Invalid failure probability.')
    if p.omega_max*t_short>1:
        raise ValueError('Short-time condition requires omega_max*t_short <= 1.')
    # B_max=0: static coefficient is already known, as noted after Eq. (101).
    t0=np.pi/(4*p.B_max) if p.B_max>0 else T_B
    if T_B<t0:
        raise ValueError('T_B must be at least t0=pi/(4 B_max).')
    J=max(0,int(np.floor(np.log2(T_B/t0))))
    times=t0*2.**np.arange(J+1) if p.B_max>0 else np.array([])
    pairs=2+len(times)
    eps_phi=phase_error(shots,failure_probability/pairs,quadrature_bias)
    Delta=2*p.C_max/p.omega_min+eps_phi
    if p.B_max>0 and (2*p.B_max*t0+Delta>=np.pi or 3*Delta>=np.pi):
        raise ValueError('Eq. (99) fails; increase shots/reduce bias or tighten priors.')
    if 2*(p.B_max+p.C_max)*t_short+eps_phi>=np.pi:
        raise ValueError('Short-time Ramsey angle may wrap.')
    rng=np.random.default_rng(seed)
    phases=[]
    measurements=[]
    for t in (t_short/2,t_short):
        data=oracle.ramsey('Z',t,shots,rng)
        measurements.append(data)
        phases.append(np.arctan2(data['S'],data['R']))
    S_hat=(4*phases[0]/t_short-phases[1]/(2*t_short))/3
    eps_S=p.C_max*p.omega_max**4*t_short**4/288+3*eps_phi/(2*t_short)
    history=[]
    B_hat=0.
    for j,t in enumerate(times):
        data=oracle.ramsey('Z',float(t),shots,rng)
        phi=float(np.arctan2(data['S'],data['R']))
        # Eq. (97) first, Eq. (98) at later scales. Not np.unwrap on doubling times.
        k=0 if j==0 else int(np.rint((2*t*B_hat-phi)/(2*np.pi)))
        B_hat=(phi+2*np.pi*k)/(2*t)
        history.append(dict(t=float(t),phi=phi,branch=k,B_hat=float(B_hat),measurement=data))
    eps_B=0. if not len(times) else Delta/(2*times[-1])
    return dict(B_hat=float(B_hat), B_hat_fit=float(B_hat), C_hat=float(S_hat-B_hat),
                B_plus_C_hat=float(S_hat),epsilon_S=float(eps_S),epsilon_B=float(eps_B),
                epsilon_C=float(eps_S+eps_B),omega_min=p.omega_min,omega_max=p.omega_max,
                B_max=p.B_max,C_max=p.C_max,phase_error_bound=eps_phi,
                failure_probability=0. if shots is None else failure_probability,
                guarantee='conditional_on_priors_and_quadrature_bias',
                backend='ideal_isolated_or_user_measurement_callback',
                max_interrogation_time=float(times[-1]) if len(times) else t_short,
                experiment_count=None if shots is None else int(2*shots*pairs),
                total_sensing_time=None if shots is None else float(2*shots*(1.5*t_short+sum(times))),
                history=history,short_measurements=measurements)

if __name__=='__main__':
    r=learn_BC()
    for k in ('B_hat','C_hat','epsilon_S','epsilon_B','epsilon_C','max_interrogation_time'):
        print(k,'=',r[k])
