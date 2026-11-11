"""Three-qubit support recovery from random Bell-sampling windows, Appendix D.
The command-line example uses a fixed Bell budget. Optional known_terms are
user estimates supported on S0.
"""
import argparse
import json
import math
from operator import index

import numpy as np
from scipy.linalg import expm
from scipy.stats import binomtest
from three_qubit_model import FREQUENCY_CASES, ThreeQubitOracle, PAULIS, LABELS, I8


DEFAULT_BELL_SHOTS_PER_WINDOW = 2 * 10**8


def bell_probabilities(U):
    if np.shape(U)!=(8,8) or not np.allclose(U.conj().T@U,I8,atol=1e-10):
        raise ValueError('Bell oracle must be an 8x8 unitary.')
    p=np.array([abs(np.trace(P@U)/8)**2 for P in PAULIS.values()])
    if abs(p.sum()-1)>1e-10:raise RuntimeError('Invalid Bell distribution.')
    return p/p.sum()


def six_bitstring_to_pauli_label(bitstring):
    """MindQuantum m5...m0 readout: system q0..q2, ancilla q3..q5."""
    if len(bitstring)!=6 or any(c not in '01' for c in bitstring):
        raise ValueError('Expected six measured bits.')
    pair={'00':'I','01':'Z','10':'X','11':'Y'}
    return ''.join(pair[bitstring[2-i]+bitstring[5-i]] for i in range(3))


def execute_six_qubit_bell_measurement(U,shots=200000,seed=1234):
    """Optional original MindQuantum circuit; imported only when requested.
    This optional SDK backend is not required for default numpy simulations.
    """
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.gates import H,X,Measure,UnivMathGate
    from mindquantum.simulator import Simulator
    bell_probabilities(U)
    circuit=Circuit()
    for q in range(3):
        circuit+=H.on(q);circuit+=X.on(q+3,q)
    circuit+=UnivMathGate('U_residual',U).on([0,1,2])
    for q in range(3):
        circuit+=X.on(q+3,q);circuit+=H.on(q)
    for q in range(6):circuit+=Measure(f'm{q}').on(q)
    raw=Simulator('mqvector',6,seed=seed).sampling(circuit,shots=shots,seed=seed)
    counts={p:0 for p in LABELS}
    for bits,count in raw.data.items():counts[six_bitstring_to_pauli_label(bits)]+=int(count)
    return counts,circuit,raw


def learn_structure(source=None,a_min=.1,m_max=3,gamma_bound=2.7,
                    derivative_bound=.45,omega_min=1.,omega_max=1.5,
                    failure_probability=.02,seed=1234,known_terms=(),
                    windows=None,shots_per_window=None):
    """Bounds are external priors; no unknown model coefficients enter design.
    Each remaining term must satisfy max(|B|,|C|)>=a_min. This is a condition,
    not a fact inferred by the code. m_max bounds unlearned term count.
    """
    source=ThreeQubitOracle() if source is None else source
    if not np.isfinite([a_min,gamma_bound,derivative_bound,omega_min,omega_max]).all():
        raise ValueError('Bounds must be finite.')
    if min(a_min,gamma_bound,omega_min)<=0 or omega_max<=omega_min or derivative_bound<0:
        raise ValueError('Invalid structure-learning priors.')
    if not isinstance(m_max,int) or not 1<=m_max<=63 or not 0<failure_probability<1:
        raise ValueError('Invalid count/failure budget.')
    known_terms=tuple(known_terms)
    S0={p.label for p in known_terms}
    if 'III' in S0 or len(S0)!=len(known_terms):raise ValueError('Invalid compensator support.')
    for term in known_terms:
        if term.label not in PAULIS or not np.isfinite([term.B,term.C,term.omega]).all():
            raise ValueError('Invalid estimated compensator term.')
    hn=sum(abs(p.B)+abs(p.C) for p in known_terms)
    ld=sum(abs(p.C*p.omega) for p in known_terms)
    M=gamma_bound+hn
    K=2*hn*M
    t=min(1/omega_max,a_min/(64*(M*M+ld+K)))
    r=max(1,math.ceil(32*derivative_bound*t/a_min))
    cT=M*M+(ld+K)/r
    if cT*t>a_min/64*(1+1e-12):raise RuntimeError('Residual short-time condition failed.')
    mu=a_min/4
    L=math.ceil(6*math.log(2*m_max/failure_probability))
    N=math.ceil(49*256/(72*mu**2*t**2)*(6*math.log(2)+math.log(2*L/failure_probability)))
    used_L=L if windows is None else windows
    used_N=N if shots_per_window is None else shots_per_window
    if any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<=0 for n in (used_L,used_N)):
        raise ValueError('Window/sample counts must be positive integers.')
    if used_N>=np.iinfo(np.int64).max:raise ValueError('Count exceeds int64 sampling limit.')
    threshold=37*mu*mu*t*t/256
    rng=np.random.default_rng(seed)
    starts=rng.uniform(0,2*np.pi/omega_min,size=used_L)
    support=set(S0);records=[]
    for a in starts:
        U=I8.copy();dt=t/r
        for k in range(r):
            left=a+k*dt
            hat=sum((p.coefficient(left)*PAULIS[p.label] for p in known_terms),np.zeros((8,8),complex))
            U=expm(1j*hat*dt)@expm(-1j*source.hamiltonian(left+.5*dt)*dt)@U
        probs=bell_probabilities(U)
        counts=rng.multinomial(used_N,probs)
        for p,count in zip(LABELS,counts):
            if p not in S0|{'III'} and count/used_N>=threshold:support.add(p)
        records.append(dict(t0=float(a),counts=counts,probabilities=probs))
    met=used_L>=L and used_N>=N
    return dict(support=sorted(support),learned_support=sorted(support),t=t,r=r,
                windows=used_L,shots_per_window=used_N,required_windows=L,
                required_shots_per_window=N,total_bell_samples=int(used_L)*int(used_N),
                threshold=threshold,known_support=sorted(S0),records=records,
                theorem_sample_budget_met=met,failure_probability=failure_probability if met else None,
                guarantee='conditional on external priors, a_min, m_max, compensation support and ideal pulses',
                backend='8D midpoint propagation and equivalent six-qubit Bell distribution',
                numerical_amplitude_error_bound=derivative_bound*t*t/(4*r))


def scan_shot_budgets(true_support, shot_budgets, *, trials=500,
                      windows=None, seed=1234, **structure_options):
    """Compare shot budgets at matched random windows, including the theory budget.

    Use true_support only to score the simulated trials.
    """
    truth = set(true_support)
    if not truth <= (PAULIS.keys() - {'III'}):
        raise ValueError('True support must contain only nonidentity three-qubit Pauli labels.')
    trials = index(trials)
    if trials <= 0:
        raise ValueError('trials must be positive.')
    budgets = sorted({index(shots) for shots in shot_budgets})
    max_shots = np.iinfo(np.int64).max
    if not budgets or any(not 0 < shots < max_shots for shots in budgets):
        raise ValueError('Provide at least one supported shot count.')
    if 'known_terms' in structure_options:
        structure_options['known_terms'] = tuple(structure_options['known_terms'])

    rng = np.random.default_rng(seed)
    trial_seeds = rng.integers(0, max_shots, size=trials)
    results = []
    for shots in [None, *budgets]:
        if results and shots == results[0]['shots_per_window']:
            continue
        failures = misses = false_positives = 0
        for trial_seed in trial_seeds:
            result = learn_structure(
                shots_per_window=shots, windows=windows,
                seed=int(trial_seed), **structure_options,
            )
            found = set(result['support'])
            misses += not truth <= found
            false_positives += not found <= truth
            failures += found != truth

        interval = binomtest(failures, trials).proportion_ci(method='exact')
        results.append(dict(
            windows=int(result['windows']),
            shots_per_window=result['shots_per_window'],
            total_bell_samples=result['total_bell_samples'],
            required_windows=result['required_windows'],
            required_shots_per_window=result['required_shots_per_window'],
            theorem_sample_budget_met=result['theorem_sample_budget_met'],
            trials=trials,
            failures=failures,
            trials_with_missed_terms=misses,
            trials_with_false_positives=false_positives,
            failure_rate=failures / trials,
            failure_rate_ci95=[interval.low, interval.high],
        ))
    return sorted(results, key=lambda row: row['shots_per_window'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simulate three-qubit structure learning.')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--shots-per-window', type=int, default=DEFAULT_BELL_SHOTS_PER_WINDOW,
                      help='Bell samples per window (default: %(default)s).')
    mode.add_argument('--theoretical-bell-budget', action='store_const', const=None,
                      dest='shots_per_window', help='Use the theoretically sufficient Bell budget.')
    mode.add_argument('--scan-shots', type=int, nargs='+',
                      help='Scan these budgets and the theoretical reference for the selected model.')
    parser.add_argument('--frequency-case', choices=FREQUENCY_CASES, default='detuned')
    parser.add_argument('--windows', type=int, help='Use this many windows instead of the theoretical count.')
    parser.add_argument('--trials', type=int, default=500, help='Monte Carlo trials per budget in scan mode.')
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    source = ThreeQubitOracle.from_frequency_case(args.frequency_case)
    if args.scan_shots is None:
        result = learn_structure(source=source, shots_per_window=args.shots_per_window,
                                 windows=args.windows, seed=args.seed)
        print({key: value for key, value in result.items() if key != 'records'})
    else:
        results = scan_shot_budgets(
            (term.label for term in source.terms), args.scan_shots,
            trials=args.trials, windows=args.windows, seed=args.seed, source=source,
        )
        print(json.dumps(results, indent=2))
