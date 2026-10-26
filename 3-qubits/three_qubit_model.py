"""Three Pauli terms; shared source, bit order, reshaping, and 8D Ramsey.
Labels are q0,q1,q2, as in the uploaded MindQuantum scripts.
Matrix order is |q2 q1 q0>: label XII -> I tensor I tensor X.
"""
from dataclasses import dataclass
from functools import lru_cache
from itertools import product, combinations
import numpy as np
from scipy.linalg import expm
from numerical_utils import validate_shots, square_integrals

I2=np.eye(2,dtype=complex)
X=np.array([[0,1],[1,0]],complex)
Y=np.array([[0,-1j],[1j,0]],complex)
Z=np.diag([1.,-1.]).astype(complex)
SINGLE=dict(I=I2,X=X,Y=Y,Z=Z)
LABELS=tuple(''.join(x) for x in product('IXYZ',repeat=3))
I8=np.eye(8,dtype=complex)

@lru_cache(None)
def pauli_string_matrix(label):
    if not isinstance(label,str) or len(label)!=3 or any(c not in SINGLE for c in label):
        raise ValueError('A Pauli label must have three I/X/Y/Z letters.')
    result=np.array([[1.]],complex)
    for c in label[::-1]: result=np.kron(result,SINGLE[c])
    result.setflags(write=False)
    return result

PAULIS={label:pauli_string_matrix(label) for label in LABELS}

@dataclass(frozen=True)
class PeriodicPauliTerm:
    label:str
    B:float
    C:float
    omega:float
    def coefficient(self,t): return self.B+self.C*np.cos(self.omega*t)

TERMS=(PeriodicPauliTerm('XII',.720,.080,1.),
       PeriodicPauliTerm('IXX',.360,.040,1.2),
       PeriodicPauliTerm('ZXY',.145,.015,1.5))

@dataclass(frozen=True)
class EstimationPriors:
    B_max:float=.8
    C_max:float=.1
    omega_min:float=1.
    omega_max:float=1.5
    def validate(self):
        if not np.isfinite(list(vars(self).values())).all() or min(self.B_max,self.C_max)<0:
            raise ValueError('Invalid coefficient prior.')
        if not 0<self.omega_min<self.omega_max:
            raise ValueError('Invalid frequency prior.')


def commutes(a,b):
    pauli_string_matrix(a);pauli_string_matrix(b)
    return sum(x!='I' and y!='I' and x!=y for x,y in zip(a,b))%2==0


def multiply_labels(a,b):
    bits={'I':0,'X':1,'Z':2,'Y':3}
    chars={v:k for k,v in bits.items()}
    return ''.join(chars[bits[x]^bits[y]] for x,y in zip(a,b))

@lru_cache(None)
def reshaping_group(target,support):
    """Minimal commuting generator set for the DISCOVERED support, not truth."""
    support=tuple(support)
    if target=='III' or target not in support:
        raise ValueError('Nonidentity target must belong to supplied learned support.')
    for lab in support: pauli_string_matrix(lab)
    others=set(support)-{target}
    if not others:return ('III',),()
    candidates=sorted((p for p in LABELS if p!='III' and commutes(p,target)),
                      key=lambda p:(sum(x!='I' for x in p),p))
    for number in range(1,min(5,len(others))+1):
        for gens in combinations(candidates,number):
            if any(not commutes(a,b) for a,b in combinations(gens,2)):continue
            if any(all(commutes(g,p) for g in gens) for p in others):continue
            group=['III']
            for g in gens:group += [multiply_labels(p,g) for p in group.copy()]
            if len(set(group))!=2**number:continue
            # Matrix verification guards bit-order and product-sign mistakes.
            for label in support:
                avg=sum(PAULIS[q]@PAULIS[label]@PAULIS[q] for q in group)/len(group)
                expected=PAULIS[label] if label==target else np.zeros((8,8),complex)
                if not np.allclose(avg,expected,atol=1e-12):
                    raise RuntimeError('Reshaping twirl failed verification.')
            return tuple(group),tuple(gens)
    raise RuntimeError('No reshaping group found for learned support.')

@lru_cache(None)
def ramsey_probe(target):
    """R anticommutes with P; S=-i P R yields cos(2theta),sin(2theta)."""
    P=PAULIS[target]
    ref=list('III')
    for i,c in enumerate(target):
        if c!='I':
            ref[i]='Z' if c in 'XY' else 'X'
            break
    else:raise ValueError('Identity has no Ramsey probe.')
    ref=''.join(ref)
    R=PAULIS[ref]
    S=-1j*P@R
    state=np.array([1.],complex)
    for c in ref[::-1]:
        ket=np.array([1.,1.],complex)/np.sqrt(2) if c=='X' else np.array([1.,0.],complex)
        state=np.kron(state,ket)
    if not np.allclose(S,S.conj().T) or not np.allclose(R@state,state):
        raise RuntimeError('Invalid generalized Ramsey construction.')
    return state,R,S,ref


def measure_state(state,target,shots,rng):
    validate_shots(shots)
    _,R,S,ref=ramsey_probe(target)
    means=np.array([np.vdot(state,O@state).real for O in (R,S)])
    if abs(np.vdot(state,state).real-1)>1e-9:raise RuntimeError('Nonunit norm.')
    counts=None
    if shots is not None:
        counts=rng.binomial(shots,np.clip((1+means)/2,0,1))
        means=2*counts/shots-1
    return dict(R=float(means[0]),S=float(means[1]),counts_plus=counts,reference=ref)

class ThreeQubitOracle:
    """Truth is confined to this simulation source and optional diagnostics.
    Ideal Ramsey uses a full 8D Pauli propagator after ideal reshaping.
    Frequency ESU is the ideal logical sensing-qubit query model.
    """
    def __init__(self,terms=TERMS):
        self.terms=tuple(terms)
        self.by_label={t.label:t for t in self.terms}
        if len(self.by_label)!=len(self.terms):raise ValueError('Duplicate model terms.')
        for t in self.terms:
            pauli_string_matrix(t.label)
            if t.label=='III' or not np.isfinite([t.B,t.C,t.omega]).all() or t.omega<=0:
                raise ValueError('Invalid model term.')
    def hamiltonian(self,t):
        return sum((term.coefficient(t)*PAULIS[term.label] for term in self.terms),np.zeros((8,8),complex))
    def isolated_phase(self,target,t):
        term=self.by_label[target]
        return term.B*t+term.C*t*np.sinc(term.omega*t/np.pi)
    def esu_phase(self,target,candidate,t,B_hat=0.,scale=1.):
        term=self.by_label[target]
        g,f=square_integrals(candidate,term.omega,t)
        return scale*((term.B-B_hat)*g+term.C*f)
    def ramsey_at_phase(self,target,theta,shots,rng):
        ket,*_=ramsey_probe(target)
        state=np.cos(theta)*ket-1j*np.sin(theta)*(PAULIS[target]@ket)
        return measure_state(state,target,shots,rng)
    def ramsey(self,target,t,shots,rng):
        return self.ramsey_at_phase(target,self.isolated_phase(target,t),shots,rng)
    def refined_ramsey(self,target,kind,m,omega_hat,shots,rng):
        t=2*np.pi*m/omega_hat
        phase=self.isolated_phase(target,t) if kind=='B' else self.esu_phase(target,omega_hat,t)
        return self.ramsey_at_phase(target,phase,shots,rng)

class PulseRamseyOracle:
    """Optional deterministic finite reshaping of the complete 8x8 H(t).
    Observable error <= 2 (L+Gamma^2)*T*h; no randomized-channel theorem assumed.
    This backend implements ordinary Ramsey, not coherent ESU/QSP attenuation.
    """
    def __init__(self,source,support,gamma_bound=2.7,derivative_bound=.45,
                 quadrature_bias=.002,cycle_dt_max=.002,max_cycles=200000):
        self.source=source;self.support=tuple(support)
        self.gamma=gamma_bound;self.derivative=derivative_bound
        self.quadrature_bias=quadrature_bias;self.cycle_dt_max=cycle_dt_max;self.max_cycles=max_cycles
        if min(gamma_bound,quadrature_bias,cycle_dt_max)<=0 or derivative_bound<0:
            raise ValueError('Invalid pulse bounds.')
    def ramsey(self,target,t,shots,rng):
        if t<=0:raise ValueError('Time must be positive.')
        group,_=reshaping_group(target,self.support)
        factor=self.derivative+self.gamma**2
        h=min(self.cycle_dt_max,self.quadrature_bias/(2*factor*t))
        n=max(1,int(np.ceil(t/h)))
        if n>self.max_cycles:raise RuntimeError('Finite-pulse cycle budget exceeded.')
        h=t/n;sub=h/len(group)
        state=ramsey_probe(target)[0].copy()
        for j in range(n):
            for k,q in enumerate(group):
                H=self.source.hamiltonian(j*h+(k+.5)*sub)
                Q=PAULIS[q]
                state=expm(-1j*(Q@H@Q)*sub)@state
        result=measure_state(state,target,shots,rng)
        result.update(cycles=n,quadrature_bias_bound=2*factor*t*h)
        return result
