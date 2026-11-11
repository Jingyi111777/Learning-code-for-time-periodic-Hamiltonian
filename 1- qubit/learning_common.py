"""Single-qubit measurement simulator. Units: hbar=1; omega in rad/time.
Truth is confined to ModelOracle. Estimators receive measurements and priors.
Default backend is IDEAL isolated evolution, not a physical pulse implementation.
"""
from dataclasses import dataclass
import math
import numpy as np

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1., -1.]).astype(complex)
PAULIS = dict(I=I, X=X, Y=Y, Z=Z)

@dataclass(frozen=True)
class Priors:
    A_max: float = 1.0
    B_max: float = .5
    C_max: float = .12
    omega_min: float = 8.5
    omega_max: float = 11.
    def validate(self):
        vals = list(vars(self).values())
        if not all(np.isfinite(vals)) or min(self.A_max,self.B_max,self.C_max)<0:
            raise ValueError('Invalid coefficient priors.')
        if not 0 < self.omega_min < self.omega_max:
            raise ValueError('Require 0 < omega_min < omega_max.')

@dataclass(frozen=True)
class ModelOracle:
    """Replace this object with experiment callbacks for actual learning."""
    A: float = .8
    B: float = .3
    C: float = .1
    omega: float = 9.8
    def hamiltonian(self, t):
        return self.A*X+(self.B+self.C*np.cos(self.omega*t))*Z
    def isolated_phase(self, target, t):
        if target == 'X':
            return self.A*t
        if target == 'Z':
            return self.B*t+self.C*t*np.sinc(self.omega*t/np.pi)
        raise ValueError('Target must be X or Z.')
    def esu_phase(self, candidate, t, B_hat=0., scale=1.):
        g,f = square_integrals(candidate, self.omega, t)
        return scale*((self.B-B_hat)*g+self.C*f)
    def ramsey(self, target, t, shots, rng):
        return quadratures(self.isolated_phase(target,t), shots, rng)
    def refined_ramsey(self, kind, m, omega_hat, shots, rng):
        t = 2*np.pi*m/omega_hat
        phase = (self.isolated_phase('Z', t) if kind == 'B' else
                 self.esu_phase(omega_hat,t))
        return quadratures(phase,shots,rng)


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

@dataclass
class PulseRamseyOracle:
    """Optional finite, deterministic {I,P} reshaping for X/Z Ramsey.
    Midpoint Hamiltonians are used in each half cycle. The conservative bound
      ||U_num-U_ideal|| <= (L+Gamma^2)*t*cycle_dt
    follows from freezing to the cycle's left endpoint, two-factor splitting,
    and telescoping. Observable bias <= 2 times this unitary bound.
    It is a deterministic bound; it does NOT invoke randomized Theorem 2.
    This class does not implement the coherent ESU oracle or attenuation.
    """
    model: ModelOracle
    priors: Priors
    quadrature_bias: float = .002
    cycle_dt_max: float = .002
    max_cycles: int = 2000000

    def ramsey(self, target, t, shots, rng):
        self.priors.validate()
        if target not in ('X','Z') or t<=0 or self.quadrature_bias<=0 or self.cycle_dt_max<=0:
            raise ValueError('Invalid pulse experiment parameters.')
        p=self.priors
        G=p.A_max+p.B_max+p.C_max
        L=p.C_max*p.omega_max
        factor=L+G*G
        h=min(self.cycle_dt_max,self.quadrature_bias/(2*factor*t))
        cycles=max(1,int(np.ceil(t/h)))
        if cycles>self.max_cycles:
            raise RuntimeError('Pulse budget exceeds max_cycles; relax bias or shorten time.')
        dt=t/cycles
        Q=PAULIS[target]
        state=np.array([1.,0.],complex) if target=='X' else np.array([1.,1.],complex)/np.sqrt(2)
        for k in range(cycles):
            for frac,conjugate in ((.25,False),(.75,True)):
                H=self.model.hamiltonian((k+frac)*dt)
                if conjugate:
                    H=Q@H@Q
                # All target and non-target terms retained, no scalar-phase shortcut.
                radius=float(np.sqrt(np.trace(H@H).real/2))
                U=(np.cos(radius*dt/2)*I-1j*(dt/2)*np.sinc(radius*dt/(2*np.pi))*H)
                state=U@state
        R=float(np.vdot(state,(Z if target=='X' else X)@state).real)
        S=float(np.vdot(state,(-Y if target=='X' else Y)@state).real)
        validate_shots(shots)
        counts=None
        if shots is not None:
            counts=rng.binomial(shots,np.clip((1+np.array([R,S]))/2,0,1))
            R,S=2*counts/shots-1
        return dict(R=float(R),S=float(S),counts_plus=counts,cycles=cycles,
                    quadrature_bias_bound=2*factor*t*dt)
