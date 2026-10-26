"""Shared sampling and phase confidence utilities."""
import math
import numpy as np

def validate_shots(shots):
    if shots is not None and (isinstance(shots,bool) or not isinstance(shots,(int,np.integer)) or shots<=0):
        raise ValueError('shots must be a positive integer or None.')

def quadratures(theta, shots, rng):
    validate_shots(shots)
    exact = np.array([np.cos(2*theta), np.sin(2*theta)])
    if shots is None:
        return dict(R=float(exact[0]),S=float(exact[1]),counts_plus=None)
    counts = rng.binomial(shots,np.clip((1+exact)/2,0,1))
    measured = 2*counts/shots-1
    return dict(R=float(measured[0]),S=float(measured[1]),counts_plus=counts)

def phase_error(shots, pair_failure, quadrature_bias=0.):
    """Bound on FULL Ramsey angle 2 theta, Eqs. (90)-(92).
    quadrature_bias must cover channel/control/integration biases per observable.
    """
    validate_shots(shots)
    if not 0 < pair_failure < 1 or quadrature_bias < 0:
        raise ValueError('Invalid phase-error budget.')
    nu = 0. if shots is None else math.sqrt(2*math.log(4/pair_failure)/shots)
    q = math.sqrt(2)*(nu+quadrature_bias)
    if q >= 1:
        raise ValueError('No informative phase guarantee: increase shots or reduce control error.')
    return math.asin(q)+1e-12

def square_integrals(candidate, omega, t):
    """Exact, piecewise ESU integral in the corrected toggling frame."""
    if not np.isfinite([candidate,omega,t]).all() or min(candidate,omega)<=0 or t<0:
        raise ValueError('Invalid frequency/time.')
    switches=(2*np.arange(int(np.ceil(candidate*t/np.pi+.5)))+1)*np.pi/(2*candidate)
    edges=np.r_[0., switches[switches<t],t]
    signs=(-1.)**np.arange(len(edges)-1)
    return (float(signs@np.diff(edges)),
            float(signs@np.diff(np.sin(omega*edges))/omega))

def interval_update(prior, phase, gain, error, max_aliases=10000):
    """Intersect with ALL theta=phase+k*pi branches; phase is a HALF angle."""
    if gain<=0 or error<0:
        raise ValueError('Invalid interval gain/error.')
    out=[]
    for lo,hi in prior:
        first=math.ceil((gain*lo-error-phase)/np.pi-1e-12)
        last=math.floor((gain*hi+error-phase)/np.pi+1e-12)
        if last-first+1>max_aliases:
            raise RuntimeError('Too many phase aliases.')
        for k in range(first,last+1):
            a=max(lo,(phase+k*np.pi-error)/gain)
            b=min(hi,(phase+k*np.pi+error)/gain)
            if a<=b:
                if out and a<=out[-1][1]+1e-14:
                    out[-1]=(out[-1][0],max(b,out[-1][1]))
                else:
                    out.append((a,b))
    if not out:
        raise RuntimeError('Empty confidence intersection: check priors/noise/model.')
    return out
