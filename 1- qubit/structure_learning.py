
import math
import numpy as np
from scipy.linalg import expm
from learning_common import ModelOracle, Priors, PAULIS, I


def learn_structure(oracle=None, priors=None, a_min=.1, m_max=3,
                    failure_probability=.02, seed=1234, known_support=(),
                    h_hat=None, shots_per_window=None, windows=None):
    p=Priors() if priors is None else priors
    p.validate()
    oracle=ModelOracle() if oracle is None else oracle
    if not np.isfinite(a_min) or a_min<=0 or not 1<=m_max<=3:
        raise ValueError('Require a_min>0 and 1<=m_max<=3.')
    if not 0<failure_probability<1:
        raise ValueError('Invalid failure probability.')
    S0=set(known_support)
    if not S0<=set('XYZ'):
        raise ValueError('Unknown support labels.')
    h_hat=np.zeros((2,2),complex) if h_hat is None else np.asarray(h_hat,complex)
    if h_hat.shape!=(2,2) or not np.allclose(h_hat,h_hat.conj().T):
        raise ValueError('h_hat must be Hermitian 2x2.')
    if abs(np.trace(h_hat))>1e-12:
        raise ValueError('Compensator must be traceless.')
    if any(abs(np.trace(P@h_hat)/2)>1e-12 for lab,P in PAULIS.items() if lab not in S0|{'I'}):
        raise ValueError('Compensator must be supported on S0.')
    # This interface supports CONSTANT previously learned compensation.
    gamma=p.A_max+p.B_max+p.C_max
    hnorm=float(np.linalg.norm(h_hat,2))
    M=gamma+hnorm
    K=2*hnorm*gamma
    derivative=p.C_max*p.omega_max
    r=1
    t=min(1/p.omega_max, a_min/(64*max(M*M+K,1e-30)))
    # Reserve half the D32 amplitude budget for midpoint simulation error.
    r=max(1,math.ceil(32*derivative*t/a_min))
    cT=M*M+K/r
    if cT*t>a_min/64*(1+1e-12):
        raise RuntimeError('Residual short-time condition failed.')
    L=math.ceil(6*math.log(2*m_max/failure_probability))
    mu=a_min/4
    threshold=37*mu**2*t**2/256
    N=math.ceil(49*256/(72*mu**2*t**2)*(2*math.log(2)+math.log(2*L/failure_probability)))
    used_L=L if windows is None else int(windows)
    used_N=N if shots_per_window is None else int(shots_per_window)
    if min(used_L,used_N)<=0 or used_N>=np.iinfo(np.int64).max:
        raise ValueError('Sample count outside supported range.')
    rng=np.random.default_rng(seed)
    starts=rng.uniform(0,2*np.pi/p.omega_min,size=used_L)
    support=set(S0)
    records=[]
    for a in starts:
        dt=t/r
        U=I.copy()
        correction=expm(1j*h_hat*dt)
        for k in range(r):
            U=correction@expm(-1j*oracle.hamiltonian(a+(k+.5)*dt)*dt)@U
        # <Bell_P|(U x I)|Phi+> = Tr(P^dagger U)/2.
        probs=np.array([abs(np.trace(P.conj().T@U)/2)**2 for P in PAULIS.values()])
        if abs(probs.sum()-1)>1e-9:
            raise RuntimeError('Bell probabilities do not sum to one.')
        counts=rng.multinomial(used_N,probs/probs.sum())
        for lab,count in zip(PAULIS,counts):
            if lab not in S0|{'I'} and count/used_N>=threshold:
                support.add(lab)
        records.append(dict(t0=float(a),probabilities=probs,counts=counts))
    return dict(support=sorted(support),t=t,r=r,windows=used_L,shots_per_window=used_N,
                required_windows=L,required_shots_per_window=N,threshold=threshold,
                theorem_sample_budget_met=used_L>=L and used_N>=N,
                failure_probability=failure_probability if used_L>=L and used_N>=N else None,
                guarantee='conditional_on_priors_a_min_m_max_and_compensation_support',
                total_bell_samples=used_L*used_N,records=records,
                simulation_amplitude_error_bound=derivative*t*t/(4*r))

if __name__=='__main__':
    r=learn_structure()
    print({k:v for k,v in r.items() if k!='records'})
