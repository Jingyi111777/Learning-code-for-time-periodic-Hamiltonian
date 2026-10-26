
from __future__ import annotations

TARGET_LABEL = "IXX"                 # 可选 XII / IXX / ZXY
LAMBDA_VALUES = [0.6, 0.2, 0.05]     # 三个需要比较的 lambda
REPEATS = 30                         # 每个 lambda 的重复次数


# ===== numerical_utils.py =====
import math
import numpy as np

def validate_shots(shots):
    if shots is not None and (isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots <= 0):
        raise ValueError('shots must be a positive integer or None.')

def quadratures(theta, shots, rng):
    validate_shots(shots)
    exact = np.array([np.cos(2 * theta), np.sin(2 * theta)])
    if shots is None:
        return dict(R=float(exact[0]), S=float(exact[1]), counts_plus=None)
    counts = rng.binomial(shots, np.clip((1 + exact) / 2, 0, 1))
    measured = 2 * counts / shots - 1
    return dict(R=float(measured[0]), S=float(measured[1]), counts_plus=counts)

def phase_error(shots, pair_failure, quadrature_bias=0.0):
    """Bound on FULL Ramsey angle 2 theta, Eqs. (90)-(92).
    quadrature_bias must cover channel/control/integration biases per observable.
    """
    validate_shots(shots)
    if not 0 < pair_failure < 1 or quadrature_bias < 0:
        raise ValueError('Invalid phase-error budget.')
    nu = 0.0 if shots is None else math.sqrt(2 * math.log(4 / pair_failure) / shots)
    q = math.sqrt(2) * (nu + quadrature_bias)
    if q >= 1:
        raise ValueError('No informative phase guarantee: increase shots or reduce control error.')
    return math.asin(q) + 1e-12

def square_integrals(candidate, omega, t):
    """Exact, piecewise ESU integral in the corrected toggling frame."""
    if not np.isfinite([candidate, omega, t]).all() or min(candidate, omega) <= 0 or t < 0:
        raise ValueError('Invalid frequency/time.')
    switches = (2 * np.arange(int(np.ceil(candidate * t / np.pi + 0.5))) + 1) * np.pi / (2 * candidate)
    edges = np.r_[0.0, switches[switches < t], t]
    signs = (-1.0) ** np.arange(len(edges) - 1)
    return (float(signs @ np.diff(edges)), float(signs @ np.diff(np.sin(omega * edges)) / omega))

def interval_update(prior, phase, gain, error, max_aliases=10000):
    """Intersect with ALL theta=phase+k*pi branches; phase is a HALF angle."""
    if gain <= 0 or error < 0:
        raise ValueError('Invalid interval gain/error.')
    out = []
    for lo, hi in prior:
        first = math.ceil((gain * lo - error - phase) / np.pi - 1e-12)
        last = math.floor((gain * hi + error - phase) / np.pi + 1e-12)
        if last - first + 1 > max_aliases:
            raise RuntimeError('Too many phase aliases.')
        for k in range(first, last + 1):
            a = max(lo, (phase + k * np.pi - error) / gain)
            b = min(hi, (phase + k * np.pi + error) / gain)
            if a <= b:
                if out and a <= out[-1][1] + 1e-14:
                    out[-1] = (out[-1][0], max(b, out[-1][1]))
                else:
                    out.append((a, b))
    if not out:
        raise RuntimeError('Empty confidence intersection: check priors/noise/model.')
    return out


# ===== three_qubit_model.py =====
from dataclasses import dataclass
from functools import lru_cache
from itertools import product, combinations
import numpy as np
from scipy.linalg import expm
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.diag([1.0, -1.0]).astype(complex)
SINGLE = dict(I=I2, X=X, Y=Y, Z=Z)
LABELS = tuple((''.join(x) for x in product('IXYZ', repeat=3)))
I8 = np.eye(8, dtype=complex)

@lru_cache(None)
def pauli_string_matrix(label):
    if not isinstance(label, str) or len(label) != 3 or any((c not in SINGLE for c in label)):
        raise ValueError('A Pauli label must have three I/X/Y/Z letters.')
    result = np.array([[1.0]], complex)
    for c in label[::-1]:
        result = np.kron(result, SINGLE[c])
    result.setflags(write=False)
    return result
PAULIS = {label: pauli_string_matrix(label) for label in LABELS}

@dataclass(frozen=True)
class PeriodicPauliTerm:
    label: str
    B: float
    C: float
    omega: float

    def coefficient(self, t):
        return self.B + self.C * np.cos(self.omega * t)
TERMS = (PeriodicPauliTerm('XII', 0.72, 0.08, 1.0), PeriodicPauliTerm('IXX', 0.36, 0.04, 1.2), PeriodicPauliTerm('ZXY', 0.145, 0.015, 1.5))

@dataclass(frozen=True)
class EstimationPriors:
    B_max: float = 0.8
    C_max: float = 0.1
    omega_min: float = 1.0
    omega_max: float = 1.5

    def validate(self):
        if not np.isfinite(list(vars(self).values())).all() or min(self.B_max, self.C_max) < 0:
            raise ValueError('Invalid coefficient prior.')
        if not 0 < self.omega_min < self.omega_max:
            raise ValueError('Invalid frequency prior.')

def commutes(a, b):
    pauli_string_matrix(a)
    pauli_string_matrix(b)
    return sum((x != 'I' and y != 'I' and (x != y) for x, y in zip(a, b))) % 2 == 0

def multiply_labels(a, b):
    bits = {'I': 0, 'X': 1, 'Z': 2, 'Y': 3}
    chars = {v: k for k, v in bits.items()}
    return ''.join((chars[bits[x] ^ bits[y]] for x, y in zip(a, b)))

@lru_cache(None)
def reshaping_group(target, support):
    """Minimal commuting generator set for the DISCOVERED support, not truth."""
    support = tuple(support)
    if target == 'III' or target not in support:
        raise ValueError('Nonidentity target must belong to supplied learned support.')
    for lab in support:
        pauli_string_matrix(lab)
    others = set(support) - {target}
    if not others:
        return (('III',), ())
    candidates = sorted((p for p in LABELS if p != 'III' and commutes(p, target)), key=lambda p: (sum((x != 'I' for x in p)), p))
    for number in range(1, min(5, len(others)) + 1):
        for gens in combinations(candidates, number):
            if any((not commutes(a, b) for a, b in combinations(gens, 2))):
                continue
            if any((all((commutes(g, p) for g in gens)) for p in others)):
                continue
            group = ['III']
            for g in gens:
                group += [multiply_labels(p, g) for p in group.copy()]
            if len(set(group)) != 2 ** number:
                continue
            for label in support:
                avg = sum((PAULIS[q] @ PAULIS[label] @ PAULIS[q] for q in group)) / len(group)
                expected = PAULIS[label] if label == target else np.zeros((8, 8), complex)
                if not np.allclose(avg, expected, atol=1e-12):
                    raise RuntimeError('Reshaping twirl failed verification.')
            return (tuple(group), tuple(gens))
    raise RuntimeError('No reshaping group found for learned support.')

@lru_cache(None)
def ramsey_probe(target):
    """R anticommutes with P; S=-i P R yields cos(2theta),sin(2theta)."""
    P = PAULIS[target]
    ref = list('III')
    for i, c in enumerate(target):
        if c != 'I':
            ref[i] = 'Z' if c in 'XY' else 'X'
            break
    else:
        raise ValueError('Identity has no Ramsey probe.')
    ref = ''.join(ref)
    R = PAULIS[ref]
    S = -1j * P @ R
    state = np.array([1.0], complex)
    for c in ref[::-1]:
        ket = np.array([1.0, 1.0], complex) / np.sqrt(2) if c == 'X' else np.array([1.0, 0.0], complex)
        state = np.kron(state, ket)
    if not np.allclose(S, S.conj().T) or not np.allclose(R @ state, state):
        raise RuntimeError('Invalid generalized Ramsey construction.')
    return (state, R, S, ref)

def measure_state(state, target, shots, rng):
    validate_shots(shots)
    _, R, S, ref = ramsey_probe(target)
    means = np.array([np.vdot(state, O @ state).real for O in (R, S)])
    if abs(np.vdot(state, state).real - 1) > 1e-09:
        raise RuntimeError('Nonunit norm.')
    counts = None
    if shots is not None:
        counts = rng.binomial(shots, np.clip((1 + means) / 2, 0, 1))
        means = 2 * counts / shots - 1
    return dict(R=float(means[0]), S=float(means[1]), counts_plus=counts, reference=ref)

class ThreeQubitOracle:
    """Truth is confined to this simulation source and optional diagnostics.
    Ideal Ramsey uses a full 8D Pauli propagator after ideal reshaping.
    Frequency ESU is the ideal logical sensing-qubit query model.
    """

    def __init__(self, terms=TERMS):
        self.terms = tuple(terms)
        self.by_label = {t.label: t for t in self.terms}
        if len(self.by_label) != len(self.terms):
            raise ValueError('Duplicate model terms.')
        for t in self.terms:
            pauli_string_matrix(t.label)
            if t.label == 'III' or not np.isfinite([t.B, t.C, t.omega]).all() or t.omega <= 0:
                raise ValueError('Invalid model term.')

    def hamiltonian(self, t):
        return sum((term.coefficient(t) * PAULIS[term.label] for term in self.terms), np.zeros((8, 8), complex))

    def isolated_phase(self, target, t):
        term = self.by_label[target]
        return term.B * t + term.C * t * np.sinc(term.omega * t / np.pi)

    def esu_phase(self, target, candidate, t, B_hat=0.0, scale=1.0):
        term = self.by_label[target]
        g, f = square_integrals(candidate, term.omega, t)
        return scale * ((term.B - B_hat) * g + term.C * f)

    def ramsey_at_phase(self, target, theta, shots, rng):
        ket, *_ = ramsey_probe(target)
        state = np.cos(theta) * ket - 1j * np.sin(theta) * (PAULIS[target] @ ket)
        return measure_state(state, target, shots, rng)

    def ramsey(self, target, t, shots, rng):
        return self.ramsey_at_phase(target, self.isolated_phase(target, t), shots, rng)

    def refined_ramsey(self, target, kind, m, omega_hat, shots, rng):
        t = 2 * np.pi * m / omega_hat
        phase = self.isolated_phase(target, t) if kind == 'B' else self.esu_phase(target, omega_hat, t)
        return self.ramsey_at_phase(target, phase, shots, rng)

class PulseRamseyOracle:
    """Optional deterministic finite reshaping of the complete 8x8 H(t).
    Observable error <= 2 (L+Gamma^2)*T*h; no randomized-channel theorem assumed.
    This backend implements ordinary Ramsey, not coherent ESU/QSP attenuation.
    """

    def __init__(self, source, support, gamma_bound=2.7, derivative_bound=0.45, quadrature_bias=0.002, cycle_dt_max=0.002, max_cycles=200000):
        self.source = source
        self.support = tuple(support)
        self.gamma = gamma_bound
        self.derivative = derivative_bound
        self.quadrature_bias = quadrature_bias
        self.cycle_dt_max = cycle_dt_max
        self.max_cycles = max_cycles
        if min(gamma_bound, quadrature_bias, cycle_dt_max) <= 0 or derivative_bound < 0:
            raise ValueError('Invalid pulse bounds.')

    def ramsey(self, target, t, shots, rng):
        if t <= 0:
            raise ValueError('Time must be positive.')
        group, _ = reshaping_group(target, self.support)
        factor = self.derivative + self.gamma ** 2
        h = min(self.cycle_dt_max, self.quadrature_bias / (2 * factor * t))
        n = max(1, int(np.ceil(t / h)))
        if n > self.max_cycles:
            raise RuntimeError('Finite-pulse cycle budget exceeded.')
        h = t / n
        sub = h / len(group)
        state = ramsey_probe(target)[0].copy()
        for j in range(n):
            for k, q in enumerate(group):
                H = self.source.hamiltonian(j * h + (k + 0.5) * sub)
                Q = PAULIS[q]
                state = expm(-1j * (Q @ H @ Q) * sub) @ state
        result = measure_state(state, target, shots, rng)
        result.update(cycles=n, quadrature_bias_bound=2 * factor * t * h)
        return result


# ===== structure_learning.py =====
import math
import numpy as np
from scipy.linalg import expm

def bell_probabilities(U):
    if np.shape(U) != (8, 8) or not np.allclose(U.conj().T @ U, I8, atol=1e-10):
        raise ValueError('Bell oracle must be an 8x8 unitary.')
    p = np.array([abs(np.trace(P @ U) / 8) ** 2 for P in PAULIS.values()])
    if abs(p.sum() - 1) > 1e-10:
        raise RuntimeError('Invalid Bell distribution.')
    return p / p.sum()

def six_bitstring_to_pauli_label(bitstring):
    """MindQuantum m5...m0 readout: system q0..q2, ancilla q3..q5."""
    if len(bitstring) != 6 or any((c not in '01' for c in bitstring)):
        raise ValueError('Expected six measured bits.')
    pair = {'00': 'I', '01': 'Z', '10': 'X', '11': 'Y'}
    return ''.join((pair[bitstring[2 - i] + bitstring[5 - i]] for i in range(3)))

def execute_six_qubit_bell_measurement(U, shots=200000, seed=1234):
    """Optional original MindQuantum circuit; imported only when requested.
    This optional SDK backend is not required for default numpy simulations.
    """
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.gates import H, X, Measure, UnivMathGate
    from mindquantum.simulator import Simulator
    bell_probabilities(U)
    circuit = Circuit()
    for q in range(3):
        circuit += H.on(q)
        circuit += X.on(q + 3, q)
    circuit += UnivMathGate('U_residual', U).on([0, 1, 2])
    for q in range(3):
        circuit += X.on(q + 3, q)
        circuit += H.on(q)
    for q in range(6):
        circuit += Measure(f'm{q}').on(q)
    raw = Simulator('mqvector', 6, seed=seed).sampling(circuit, shots=shots, seed=seed)
    counts = {p: 0 for p in LABELS}
    for bits, count in raw.data.items():
        counts[six_bitstring_to_pauli_label(bits)] += int(count)
    return (counts, circuit, raw)

def learn_structure(source=None, a_min=0.1, m_max=3, gamma_bound=2.7, derivative_bound=0.45, omega_min=1.0, omega_max=1.5, failure_probability=0.02, seed=1234, known_terms=(), windows=None, shots_per_window=None):
    """Bounds are external priors; no unknown model coefficients enter design.
    Each remaining term must satisfy max(|B|,|C|)>=a_min. This is a condition,
    not a fact inferred by the code. m_max bounds unlearned term count.
    """
    source = ThreeQubitOracle() if source is None else source
    if not np.isfinite([a_min, gamma_bound, derivative_bound, omega_min, omega_max]).all():
        raise ValueError('Bounds must be finite.')
    if min(a_min, gamma_bound, omega_min) <= 0 or omega_max <= omega_min or derivative_bound < 0:
        raise ValueError('Invalid structure-learning priors.')
    if not isinstance(m_max, int) or not 1 <= m_max <= 63 or (not 0 < failure_probability < 1):
        raise ValueError('Invalid count/failure budget.')
    known_terms = tuple(known_terms)
    S0 = {p.label for p in known_terms}
    if 'III' in S0 or len(S0) != len(known_terms):
        raise ValueError('Invalid compensator support.')
    for term in known_terms:
        if term.label not in PAULIS or not np.isfinite([term.B, term.C, term.omega]).all():
            raise ValueError('Invalid estimated compensator term.')
    hn = sum((abs(p.B) + abs(p.C) for p in known_terms))
    ld = sum((abs(p.C * p.omega) for p in known_terms))
    M = gamma_bound + hn
    K = 2 * hn * M
    t = min(1 / omega_max, a_min / (64 * (M * M + ld + K)))
    r = max(1, math.ceil(32 * derivative_bound * t / a_min))
    cT = M * M + (ld + K) / r
    if cT * t > a_min / 64 * (1 + 1e-12):
        raise RuntimeError('Residual short-time condition failed.')
    mu = a_min / 4
    L = math.ceil(6 * math.log(2 * m_max / failure_probability))
    N = math.ceil(49 * 256 / (72 * mu ** 2 * t ** 2) * (6 * math.log(2) + math.log(2 * L / failure_probability)))
    used_L = L if windows is None else windows
    used_N = N if shots_per_window is None else shots_per_window
    if any((isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n <= 0 for n in (used_L, used_N))):
        raise ValueError('Window/sample counts must be positive integers.')
    if used_N >= np.iinfo(np.int64).max:
        raise ValueError('Count exceeds int64 sampling limit.')
    threshold = 37 * mu * mu * t * t / 256
    rng = np.random.default_rng(seed)
    starts = rng.uniform(0, 2 * np.pi / omega_min, size=used_L)
    support = set(S0)
    records = []
    for a in starts:
        U = I8.copy()
        dt = t / r
        for k in range(r):
            left = a + k * dt
            hat = sum((p.coefficient(left) * PAULIS[p.label] for p in known_terms), np.zeros((8, 8), complex))
            U = expm(1j * hat * dt) @ expm(-1j * source.hamiltonian(left + 0.5 * dt) * dt) @ U
        probs = bell_probabilities(U)
        counts = rng.multinomial(used_N, probs)
        for p, count in zip(LABELS, counts):
            if p not in S0 | {'III'} and count / used_N >= threshold:
                support.add(p)
        records.append(dict(t0=float(a), counts=counts, probabilities=probs))
    met = used_L >= L and used_N >= N
    return dict(support=sorted(support), learned_support=sorted(support), t=t, r=r, windows=used_L, shots_per_window=used_N, required_windows=L, required_shots_per_window=N, total_bell_samples=int(used_L * used_N), threshold=threshold, known_support=sorted(S0), records=records, theorem_sample_budget_met=met, failure_probability=failure_probability if met else None, guarantee='conditional on external priors, a_min, m_max, compensation support and ideal pulses', backend='8D midpoint propagation and equivalent six-qubit Bell distribution', numerical_amplitude_error_bound=derivative_bound * t * t / (4 * r))

Priors = EstimationPriors
ModelOracle = ThreeQubitOracle


# ===== estimation_core.py =====
import math
import numpy as np

def learn_one_target(oracle=None, priors=None, T_B=128.0, t_short=0.2, shots=None, failure_probability=0.02, quadrature_bias=0.0, seed=123, target_label='XII'):
    oracle = ModelOracle() if oracle is None else oracle
    p = Priors() if priors is None else priors
    p.validate()
    if not np.isfinite([T_B, t_short, failure_probability]).all() or min(T_B, t_short) <= 0:
        raise ValueError('Times must be positive and finite.')
    if not 0 < failure_probability < 1:
        raise ValueError('Invalid failure probability.')
    if p.omega_max * t_short > 1:
        raise ValueError('Short-time condition requires omega_max*t_short <= 1.')
    t0 = np.pi / (4 * p.B_max) if p.B_max > 0 else T_B
    if T_B < t0:
        raise ValueError('T_B must be at least t0=pi/(4 B_max).')
    J = max(0, int(np.floor(np.log2(T_B / t0))))
    times = t0 * 2.0 ** np.arange(J + 1) if p.B_max > 0 else np.array([])
    pairs = 2 + len(times)
    eps_phi = phase_error(shots, failure_probability / pairs, quadrature_bias)
    Delta = 2 * p.C_max / p.omega_min + eps_phi
    if p.B_max > 0 and (2 * p.B_max * t0 + Delta >= np.pi or 3 * Delta >= np.pi):
        raise ValueError('Eq. (99) fails; increase shots/reduce bias or tighten priors.')
    if 2 * (p.B_max + p.C_max) * t_short + eps_phi >= np.pi:
        raise ValueError('Short-time Ramsey angle may wrap.')
    rng = np.random.default_rng(seed)
    phases = []
    measurements = []
    for t in (t_short / 2, t_short):
        data = oracle.ramsey(target_label, t, shots, rng)
        measurements.append(data)
        phases.append(np.arctan2(data['S'], data['R']))
    S_hat = (4 * phases[0] / t_short - phases[1] / (2 * t_short)) / 3
    eps_S = p.C_max * p.omega_max ** 4 * t_short ** 4 / 288 + 3 * eps_phi / (2 * t_short)
    history = []
    B_hat = 0.0
    for j, t in enumerate(times):
        data = oracle.ramsey(target_label, float(t), shots, rng)
        phi = float(np.arctan2(data['S'], data['R']))
        k = 0 if j == 0 else int(np.rint((2 * t * B_hat - phi) / (2 * np.pi)))
        B_hat = (phi + 2 * np.pi * k) / (2 * t)
        history.append(dict(t=float(t), phi=phi, branch=k, B_hat=float(B_hat), measurement=data))
    eps_B = 0.0 if not len(times) else Delta / (2 * times[-1])
    return dict(B_hat=float(B_hat), B_hat_fit=float(B_hat), C_hat=float(S_hat - B_hat), B_plus_C_hat=float(S_hat), epsilon_S=float(eps_S), epsilon_B=float(eps_B), epsilon_C=float(eps_S + eps_B), omega_min=p.omega_min, omega_max=p.omega_max, B_max=p.B_max, C_max=p.C_max, phase_error_bound=eps_phi, failure_probability=0.0 if shots is None else failure_probability, guarantee='conditional_on_priors_and_quadrature_bias', backend='ideal_isolated_or_user_measurement_callback', max_interrogation_time=float(times[-1]) if len(times) else t_short, experiment_count=None if shots is None else int(2 * shots * pairs), total_sensing_time=None if shots is None else float(2 * shots * (1.5 * t_short + sum(times))), history=history, short_measurements=measurements)


# ===== bc_estimation.py =====
def learn_B_and_C_for_target(target_label, support, source=None, priors=None, T_B=128.0, t_short=0.2, shots=None, failure_probability=0.02, quadrature_bias=0.0, seed=123):
    source = ThreeQubitOracle() if source is None else source
    priors = EstimationPriors() if priors is None else priors
    group, generators = reshaping_group(target_label, tuple(support))
    result = learn_one_target(oracle=source, priors=priors, T_B=T_B, t_short=t_short, shots=shots, failure_probability=failure_probability, quadrature_bias=quadrature_bias, seed=seed, target_label=target_label)
    result.update(target_label=target_label, reshaping_group=group, reshaping_generators=generators, ramsey_times=[h['t'] for h in result['history']], estimation_method='Richardson S plus doubled-time Ramsey B')
    result.pop('B_hat_fit', None)
    return result

def learn_B_and_C_for_all_targets(support=None, source=None, priors=None, failure_probability=0.02, seed=123, **options):
    source = ThreeQubitOracle() if source is None else source
    if support is None:
        support = learn_structure(source)['support']
    support = tuple(support)
    if not support or len(set(support)) != len(support):
        raise ValueError('Invalid learned support.')
    return {label: learn_B_and_C_for_target(label, support, source, priors, failure_probability=failure_probability / len(support), seed=seed + 1000 * i, **options) for i, label in enumerate(support)}


# ===== omega_adaptive.py =====
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple
import math
import numpy as np
from scipy.optimize import brentq, least_squares
I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.diag([1.0, -1.0]).astype(complex)
DEFAULT_ORACLE = ThreeQubitOracle()
EXTRA_B_ERROR = 0.0
EXTRA_C_ERROR = 0.0

@dataclass(frozen=True)
class SearchDesign:
    B_hat: float
    C_hat: float
    epsilon_B: float
    epsilon_C: float
    C_minus: float
    C_plus: float
    scale: float
    T_total: float
    correction_bound: float
    alpha_min: float
    alpha_max: float
    a_max: float
    a_out: float
    a_in: float
    r_in: float
    r_out: float
    omega_min: float
    omega_max: float
    number_of_bins: int
    filter_epsilon: float
    failure_probability: float

def choose_search_design(coefficients: Mapping[str, float], omega_min: float=1.0, omega_max: float=1.5, failure_probability: float=0.05, correction_target: float=0.02, extra_B_error: float=EXTRA_B_ERROR, extra_C_error: float=EXTRA_C_ERROR, maximum_bins: int=4096, fixed_scale: float | None=None) -> SearchDesign:
    """只使用估计值、误差区间和频率先验，逐项选择四个判别参数。

    A = alpha*sinc((omega-omega_k)*T) + E,
    alpha in [pi*C_minus/(2*C_plus), pi/2], |E| <= correction_bound.
    加入余项后设计内外区阈值；旁瓣界保证外区不是只在主瓣上成立。
    """
    bh = float(coefficients.get('B_hat', coefficients.get('B_hat_fit', np.nan)))
    ch = float(coefficients['C_hat'])
    eb = float(coefficients['epsilon_B']) + extra_B_error
    ec = float(coefficients['epsilon_C']) + extra_C_error
    if not np.all(np.isfinite([bh, ch, eb, ec])) or min(eb, ec) < 0:
        raise ValueError('B/C 估计与误差界必须有限，误差半宽必须非负。')
    if not np.isfinite([omega_min, omega_max]).all() or not 0 < omega_min < omega_max < 3 * omega_min:
        raise ValueError('当前谐波余项界要求 0 < omega_min < omega_max < 3*omega_min。')
    if not 0 < failure_probability <= 0.5 or not 0 < correction_target <= 0.05:
        raise ValueError('failure_probability 应在 (0,.5]，correction_target 在 (0,.05]。')
    cminus, cplus = (abs(ch) - ec, abs(ch) + ec)
    if cminus <= 0 or cplus >= 2 * cminus:
        raise ValueError('幅度区间不够紧：需要 |C_hat|-epsilon_C>0 且 C_plus<2*C_minus。请提高上游 B/C 精度；不能强行截断区间。')
    gamma = 3 - omega_max / omega_min
    t_unscaled = np.pi ** 2 / (4 * cplus)
    e_unscaled = np.pi / (2 * omega_min * t_unscaled) * (1 + 1 / (3 * gamma)) + np.pi * eb / (2 * omega_min)
    if fixed_scale is None:
        scale = min(1.0, correction_target / e_unscaled)
    else:
        if not np.isfinite(fixed_scale) or not 0 < fixed_scale <= 1:
            raise ValueError('fixed_scale must satisfy 0 < lambda <= 1.')
        scale = float(fixed_scale)
        if scale * e_unscaled > 0.05 * (1 + 1e-12):
            raise ValueError(f'lambda={scale:g} gives correction={scale * e_unscaled:.6g}>0.05; use lambda <= {min(1.0, 0.05 / e_unscaled):.6g}. Requested lambda was not changed.')
    t = t_unscaled / scale
    corr = scale * e_unscaled
    amin, amax = (np.pi * cminus / (2 * cplus), np.pi / 2)
    outer_floor = amax * 0.218 + corr
    gap = amin - corr - outer_floor
    if gap <= 0:
        raise ValueError('余项过大，无法得到非空判别间隙。')
    aout = outer_floor + 0.1 * gap
    ain = outer_floor + 0.85 * gap
    sinc = lambda x: float(np.sinc(x / np.pi))
    rin = 0.9 * brentq(lambda x: amin * sinc(x) - corr - ain, 1e-12, np.pi)
    rout = min(np.pi, 1.02 * brentq(lambda x: amax * sinc(x) + corr - aout, 1e-12, np.pi))
    required = max(2, math.ceil((omega_max - omega_min) * t / (2 * rin)))
    n = 1 << (required - 1).bit_length()
    if not isinstance(maximum_bins, int) or maximum_bins < 2:
        raise ValueError('maximum_bins 必须是至少为 2 的整数。')
    if n > maximum_bins:
        raise ValueError(f'需要 {n} 个候选点，超过 maximum_bins={maximum_bins}。')
    return SearchDesign(bh, ch, eb, ec, cminus, cplus, scale, t, corr, amin, amax, amax + corr, aout, ain, rin, rout, omega_min, omega_max, n, failure_probability / (2 * n), failure_probability)

def build_frequency_grid(design: SearchDesign) -> np.ndarray:
    width = design.omega_max - design.omega_min
    step = width / design.number_of_bins
    if step * design.T_total > 2 * design.r_in * (1 + 1e-12):
        raise RuntimeError('频率网格未覆盖内标记区。')
    return design.omega_min + (np.arange(design.number_of_bins) + 0.5) * step

def esu_pulse_times(omega_k: float, T_total: float) -> np.ndarray:
    if omega_k <= 0 or T_total <= 0:
        raise ValueError('频率与演化时间必须为正。')
    n = np.arange(max(0, int(np.ceil(omega_k * T_total / np.pi + 0.5))))
    times = (2 * n + 1) * np.pi / (2 * omega_k)
    return times[times < T_total]

def esu_filter_integrals(omega_k: float, omega_true: float, T_total: float) -> Tuple[float, float]:
    """在方波跳变点分段，精确积分；不采用主瓣近似生成模拟数据。"""
    boundaries = np.r_[0.0, esu_pulse_times(omega_k, T_total), T_total]
    signs = (-1.0) ** np.arange(len(boundaries) - 1)
    g = np.dot(signs, np.diff(boundaries))
    f = np.dot(signs, np.diff(np.sin(omega_true * boundaries))) / omega_true
    return (float(g), float(f))

def sensing_phases_for_target(target_label: str, omega_bins: Sequence[float], design: SearchDesign) -> np.ndarray:
    """模拟未知信号源；真值仅在信号源内部使用。"""
    return np.array([DEFAULT_ORACLE.esu_phase(target_label, w, design.T_total, design.B_hat, design.scale) for w in omega_bins])

def x_phase_matrix(phi: float) -> np.ndarray:
    return np.cos(phi) * I2 + 1j * np.sin(phi) * X

def build_qsp_oracle_blocks(theta_values: Sequence[float], qsp_phases: Sequence[float]) -> np.ndarray:
    theta = np.asarray(theta_values, float)
    u = np.broadcast_to(x_phase_matrix(qsp_phases[0]), (len(theta), 2, 2)).copy()
    w = np.stack([np.exp(-1j * theta), np.exp(1j * theta)], axis=1)
    for phi in qsp_phases[1:]:
        u *= w[:, :, None]
        u = np.einsum('ij,kjl->kil', x_phase_matrix(phi), u)
    return u

def qsp_2x2_unitary(theta: float, qsp_phases: Sequence[float]) -> np.ndarray:
    return build_qsp_oracle_blocks([theta], qsp_phases)[0]

def _qsp_fit_residual_jacobian(phases: np.ndarray, theta: np.ndarray, n_stop: int):
    v = np.zeros((len(theta), 2), complex)
    v[:, 0] = 1
    derivative = np.zeros((len(theta), 2, len(phases)), complex)
    w = np.stack([np.exp(-1j * theta), np.exp(1j * theta)], axis=1)
    for k, phi in enumerate(phases):
        if k:
            v *= w
            derivative *= w[:, :, None]
        c, s = (np.cos(phi), np.sin(phi))
        v = c * v + 1j * s * v[:, ::-1]
        derivative = c * derivative + 1j * s * derivative[:, ::-1, :]
        derivative[:, :, k] += 1j * v[:, ::-1]
    error = np.r_[v[:n_stop, 1], v[n_stop:, 0]]
    jac = np.concatenate([derivative[:n_stop, 1, :], derivative[n_stop:, 0, :]])
    return (np.r_[error.real, error.imag], np.r_[jac.real, jac.imag])

def validate_qsp_filter(phases: Sequence[float], design: SearchDesign) -> dict:
    """以酉矩阵导数范数 <= d 为连续区间补足网格验证。"""
    d = len(phases) - 1
    step_max = np.sqrt(design.filter_epsilon) / (8 * max(d, 1))
    worst = 0.0
    for lo, hi, row in [(-design.a_out, design.a_out, 1), (design.a_in, design.a_max, 0), (-design.a_max, -design.a_in, 0)]:
        n = max(2, math.ceil((hi - lo) / step_max) + 1)
        theta = np.linspace(lo, hi, n)
        max_amplitude = 0.0
        for start in range(0, n, 10000):
            u = build_qsp_oracle_blocks(theta[start:start + 10000], phases)
            max_amplitude = max(max_amplitude, float(np.max(abs(u[:, row, 0]))))
        amplitude_bound = max_amplitude + d * (hi - lo) / (2 * (n - 1)) + 1e-12
        worst = max(worst, amplitude_bound ** 2)
    return {'continuous_error_bound': worst, 'filter_epsilon': design.filter_epsilon, 'passed': worst <= design.filter_epsilon, 'certificate_scope': 'Lipschitz grid certificate in floating-point arithmetic'}
QSP_PHASE_CACHE = {}

def _design_cache_key(design: SearchDesign) -> tuple:
    return (design.a_out, design.a_in, design.a_max, design.filter_epsilon)

def synthesize_qsp_phases(design: SearchDesign, seed: int=321, verbose: bool=True) -> Tuple[np.ndarray, dict]:
    key = _design_cache_key(design)
    if key in QSP_PHASE_CACHE:
        phases = np.asarray(QSP_PHASE_CACHE[key], float)
        check = validate_qsp_filter(phases, design)
        if check['passed']:
            return (phases, check)
    theta = np.r_[np.linspace(0, design.a_out, 100), np.linspace(design.a_in, design.a_max, 100)]
    rng = np.random.default_rng(seed)
    for degree in (9, 17, 25, 33, 49, 65, 81):
        for attempt in range(2):
            last_x = last_value = None

            def evaluate(x):
                nonlocal last_x, last_value
                if last_x is None or not np.array_equal(x, last_x):
                    last_x = x.copy()
                    last_value = _qsp_fit_residual_jacobian(x, theta, 100)
                return last_value
            result = least_squares(lambda p: evaluate(p)[0], rng.normal(0, 0.7, degree + 1), jac=lambda p: evaluate(p)[1], max_nfev=450, ftol=1e-10, xtol=1e-10, gtol=1e-10)
            residual = evaluate(result.x)[0]
            sampled_error = float(np.max(residual[:200] ** 2 + residual[200:] ** 2))
            if verbose:
                print(f'  QSP degree={degree:2d}, attempt={attempt + 1}, grid error={sampled_error:.3e}', flush=True)
            if sampled_error < design.filter_epsilon * 0.65:
                check = validate_qsp_filter(result.x, design)
                if check['passed']:
                    QSP_PHASE_CACHE[key] = result.x.tolist()
                    return (result.x, check)
    raise RuntimeError('QSP 相位合成未通过误差检查；没有执行未经验证的 BBHT。')

def apply_qsp_oracle(state: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    return np.einsum('kij,kj->ki', blocks, state)

def apply_good_reflection(state: np.ndarray) -> np.ndarray:
    out = state.copy()
    out[:, 1] *= -1
    return out

def apply_initial_reflection(state: np.ndarray) -> np.ndarray:
    out = state.copy()
    out[:, 0] -= 2 * np.mean(out[:, 0])
    return out

def bbht_amplitude_search(blocks: np.ndarray, degree: int, seed: int=42, lambda_bbht: float=1.2, max_bbht_rounds: int=150) -> dict:
    """G = -U (I-2|uniform,0><uniform,0|) U^dagger (I-2 Pi)。
    接受条件仅为辅助比特测得 1；不读取真值、不预先列举 marked_indices。
    """
    if not 1 < lambda_bbht < 4 / 3:
        raise ValueError('BBHT 增长因子必须满足 1 < lambda_bbht < 4/3。')
    n = len(blocks)
    if n < 1 or degree < 1 or max_bbht_rounds < 1:
        raise ValueError('Invalid BBHT dimensions or runtime cap.')
    rng = np.random.default_rng(seed)
    dagger = blocks.conj().transpose(0, 2, 1)
    initial = np.zeros((n, 2), complex)
    initial[:, 0] = 1 / np.sqrt(n)
    m = 1.0
    calls = 0
    for round_number in range(1, max_bbht_rounds + 1):
        j = int(rng.integers(0, max(1, math.ceil(m))))
        state = apply_qsp_oracle(initial, blocks)
        calls += 1
        for _ in range(j):
            state = apply_good_reflection(state)
            state = apply_qsp_oracle(state, dagger)
            state = apply_initial_reflection(state)
            state = -apply_qsp_oracle(state, blocks)
            calls += 2
        probabilities = np.abs(state) ** 2
        norm = float(probabilities.sum())
        if abs(norm - 1) > 1e-08:
            raise RuntimeError(f'酉演化归一化错误: {norm}')
        sampled = int(rng.choice(2 * n, p=(probabilities / norm).reshape(-1)))
        k, ancilla = divmod(sampled, 2)
        if ancilla == 1:
            return {'k_hat': k, 'bbht_rounds': round_number, 'grover_iterations_last_round': j, 'total_oracle_calls': calls, 'esu_queries': calls * degree}
        m = min(lambda_bbht * m, math.sqrt(n))
    raise RuntimeError('BBHT 达到轮数上限而未测到辅助比特 1；不返回伪结果。')

def run_qsp_bbht_frequency_circuit(target_label: str, coefficients: Mapping[str, float], omega_min: float=1.0, omega_max: float=1.5, failure_probability: float=0.05, seed: int=42, verbose: bool=True, sensing_callback=None, diagnostic_source=None, **design_options) -> dict:
    design = choose_search_design(coefficients, omega_min, omega_max, failure_probability, **design_options)
    bins = build_frequency_grid(design)
    if verbose:
        print(f'\n{target_label}: T={design.T_total:.6f}, scale={design.scale:.6f}, K={len(bins)}, (r_in,r_out,a_in,a_out)=({design.r_in:.6f},{design.r_out:.6f},{design.a_in:.6f},{design.a_out:.6f})', flush=True)
    phases, check = synthesize_qsp_phases(design, verbose=verbose)
    theta = sensing_phases_for_target(target_label, bins, design) if sensing_callback is None else np.asarray(sensing_callback(bins, design), float)
    if theta.shape != bins.shape or not np.isfinite(theta).all():
        raise ValueError('Invalid ESU response shape or nonfinite phases.')
    blocks = build_qsp_oracle_blocks(theta, phases)
    search = bbht_amplitude_search(blocks, len(phases) - 1, seed=seed)
    estimate = float(bins[search['k_hat']])
    diagnostic_source = DEFAULT_ORACLE if diagnostic_source is None else diagnostic_source
    term = diagnostic_source.by_label[target_label]
    x = abs(bins - term.omega) * design.T_total
    inner, outer = (x <= design.r_in, x >= design.r_out)
    good_prob = abs(blocks[:, 1, 0]) ** 2
    p_sum = float(good_prob.mean())
    conditional_bad = float(good_prob[outer].sum() / good_prob.sum()) if good_prob.sum() > 0 else 1.0
    interval_ok = abs(term.B - design.B_hat) <= design.epsilon_B and design.C_minus <= abs(term.C) <= design.C_plus
    promise_ok = np.any(inner) and np.all(abs(theta[inner]) >= design.a_in - 1e-10) and np.all(abs(theta[outer]) <= design.a_out + 1e-10) and np.all(abs(theta) <= design.a_max + 1e-10)
    error = abs(estimate - term.omega)
    radius = error * design.T_total
    region = 'inner' if radius <= design.r_in else 'gap' if radius < design.r_out else 'outer'
    return dict(target_label=target_label, design=design, omega_hat=estimate, true_omega=term.omega, absolute_error=error, frequency_error_bound=design.r_out / design.T_total, search_failure_probability_bound=failure_probability, upstream_failure_probability=coefficients.get('failure_probability', None), conditional_on='valid upstream intervals, ideal coherent ESU and filter certificate', backend='classical_statevector_of_ideal_coherent_ESU_QSP_BBHT', total_sensing_time=search['esu_queries'] * design.T_total, returned_region=region, omega_bins=bins, theta_values=theta, qsp_phases=phases, qsp_degree=len(phases) - 1, filter_check=check, p_sum=p_sum, conditional_bad_probability=conditional_bad, source_intervals_cover_truth=interval_ok, actual_response_passes_promises=bool(promise_ok), **search)

def learn_omega_for_all_targets(learned_coefficients=None, omega_min=1.0, omega_max=1.5, failure_probability=0.05, seed=42, verbose=True, source=None, **design_options):
    """failure_probability is a JOINT budget for all supplied target terms."""
    source = DEFAULT_ORACLE if source is None else source
    if learned_coefficients is None:
        learned_coefficients = learn_B_and_C_for_all_targets(source=source)
    if not learned_coefficients:
        raise ValueError('No target coefficients supplied.')
    results = {}
    for i, (label, coeff) in enumerate(learned_coefficients.items()):
        callback = lambda bins, d, lab=label: np.array([source.esu_phase(lab, w, d.T_total, d.B_hat, d.scale) for w in bins])
        results[label] = run_qsp_bbht_frequency_circuit(label, coeff, omega_min, omega_max, failure_probability / len(learned_coefficients), seed + 1000 * i, verbose, sensing_callback=callback, diagnostic_source=source, **design_options)
    return results

def print_results(results):
    print('Pauli   omega_hat       error_bound      lambda       T_ESU       QSP_degree')
    for lab, r in results.items():
        d = r['design']
        print(f"{lab:4s} {r['omega_hat']:14.9f} {r['frequency_error_bound']:14.6e} {d.scale:11.6f} {d.T_total:12.6f} {r['qsp_degree']:8d}")
        print('  filter certificate:', r['filter_check']['passed'], 'input intervals:', r['source_intervals_cover_truth'], 'ESU promises:', r['actual_response_passes_promises'])


# ===== bc_refinement.py =====
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence
import math
import numpy as np
DEFAULT_ORACLE = ThreeQubitOracle()

@dataclass(frozen=True)
class Config:
    target_B_error: float = 0.0001
    target_C_error: float = 0.0001
    max_B_time: float = 1024.0
    max_C_time: float = 64.0
    max_C_model_phase_error: float = 0.2
    shots: Optional[int] = None
    measurement_failure_probability: float = 0.01
    extra_phase_error: float = 0.0
    max_alias_candidates: int = 10000
    seed: int = 20261012

def normalize_input(record: Mapping) -> dict:
    """既接收固定输入，也接收上一版 omega 程序的原始结果记录。"""
    if 'design' in record:
        d = record['design']
        get = (lambda k: d[k]) if isinstance(d, Mapping) else lambda k: getattr(d, k)
        values = {k: float(get(k)) for k in ('B_hat', 'C_hat', 'epsilon_B', 'epsilon_C', 'omega_min', 'omega_max')}
        values.update(omega_hat=float(record['omega_hat']), frequency_error_bound=float(record['frequency_error_bound']))
    else:
        values = {k: float(record[k]) for k in ('B_hat', 'C_hat', 'epsilon_B', 'epsilon_C', 'omega_hat', 'frequency_error_bound', 'omega_min', 'omega_max')}
    if not all((np.isfinite(v) for v in values.values())):
        raise ValueError('所有输入必须为有限数值。')
    if min(values['epsilon_B'], values['epsilon_C'], values['frequency_error_bound']) < 0:
        raise ValueError('误差界必须非负。')
    wh, ew = (values['omega_hat'], values['frequency_error_bound'])
    wlo = max(values['omega_min'], wh - ew)
    whi = min(values['omega_max'], wh + ew)
    if not 0 < values['omega_min'] < values['omega_max'] or wh <= 0 or wlo <= 0 or (whi < wlo):
        raise ValueError('频率先验与估计区间不相容，或含非正频率。')
    values.update(omega_lower=wlo, omega_upper=whi, omega_uncertainty=max(abs(wlo - wh), abs(whi - wh)), C_abs_upper=abs(values['C_hat']) + values['epsilon_C'])
    return values

def systematic_phase_bound(kind: str, t: float, p: Mapping) -> float:
    """返回 theta（半 Ramsey 相位）的模型误差上界，非真误差。"""
    c, ew = (p['C_abs_upper'], p['omega_uncertainty'])
    if kind == 'B':
        return c / p['omega_lower'] * min(ew * t, 1.0)
    if kind == 'C':
        coefficient_bias = (np.pi / 2 - 1) * c * ew / p['omega_hat'] + np.pi * c / 12 * (ew * t) ** 2
        return 2 * t / np.pi * coefficient_bias
    raise ValueError('kind 必须为 B 或 C。')

def synchronized_schedule(kind: str, p: Mapping, cfg: Config) -> list[int]:
    period = 2 * np.pi / p['omega_hat']
    cap = cfg.max_B_time if kind == 'B' else cfg.max_C_time
    max_m = int(np.floor(cap / period + 1e-12))
    if kind == 'C':
        left, right = (0, max_m)
        while left < right:
            middle = (left + right + 1) // 2
            if systematic_phase_bound('C', middle * period, p) <= cfg.max_C_model_phase_error:
                left = middle
            else:
                right = middle - 1
        max_m = left
    if max_m < 1:
        return []
    periods = []
    m = 1
    while m <= max_m:
        periods.append(m)
        m *= 2
    if periods[-1] != max_m:
        periods.append(max_m)
    return periods

def square_wave_integrals(omega_hat: float, omega: float, m: int) -> tuple[float, float]:
    """真实模拟信号积分；脉冲时间只由已学习 omega_hat 决定。"""
    t = 2 * np.pi * m / omega_hat
    switches = (2 * np.arange(2 * m) + 1) * np.pi / (2 * omega_hat)
    edges = np.r_[0.0, switches, t]
    signs = (-1.0) ** np.arange(len(edges) - 1)
    g = float(np.dot(signs, np.diff(edges)))
    f = float(np.dot(signs, np.diff(np.sin(omega * edges))) / omega)
    return (g, f)

def simulate_ramsey_experiment(label: str, kind: str, m: int, omega_hat: float, shots: Optional[int], rng: np.random.Generator, source=None) -> dict:
    """可替换为实验接口。只返回测量期望/计数，不向估计器提供真实相位。"""
    source = DEFAULT_ORACLE if source is None else source
    return source.refined_ramsey(label, kind, m, omega_hat, shots, rng)

def phase_from_measurement(measurement: Mapping, shots: Optional[int], per_pair_failure: float) -> tuple[float, float]:
    phase = 0.5 * math.atan2(measurement['S'], measurement['R'])
    if shots is None:
        return (phase, 1e-12)
    eta = math.sqrt(2 * math.log(4 / per_pair_failure) / shots)
    radius = math.sqrt(2) * eta
    phase_error = np.pi / 2 if radius >= 1 else 0.5 * math.asin(radius)
    return (phase, float(phase_error))

def merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    result = []
    for lo, hi in sorted(intervals):
        if result and lo <= result[-1][1] + 1e-14:
            result[-1] = (result[-1][0], max(result[-1][1], hi))
        else:
            result.append((float(lo), float(hi)))
    return result

def intersect_phase_candidates(prior: Sequence[tuple[float, float]], phase: float, gain: float, error: float, max_aliases: int):
    """theta = gain*coefficient + bounded error (mod pi).
    枚举所有与先验相交的分支；绝不选取最接近真实系数的分支。
    """
    intervals = []
    count = 0
    for lo, hi in prior:
        first = math.ceil((gain * lo - error - phase) / np.pi - 1e-12)
        last = math.floor((gain * hi + error - phase) / np.pi + 1e-12)
        count += max(0, last - first + 1)
        if count > max_aliases:
            raise RuntimeError('相位别名过多；需要更紧先验或更短起始时间。')
        for k in range(first, last + 1):
            left = max(lo, (phase + k * np.pi - error) / gain)
            right = min(hi, (phase + k * np.pi + error) / gain)
            if left <= right:
                intervals.append((left, right))
    return merge_intervals(intervals)

def interval_estimate(intervals):
    lower, upper = (intervals[0][0], intervals[-1][1])
    estimate = 0.5 * (lower + upper)
    return (estimate, 0.5 * (upper - lower))

def refine_one_coefficient(label: str, kind: str, p: Mapping, cfg: Config, periods: Sequence[int], per_pair_failure: float, rng: np.random.Generator, source=None) -> dict:
    center, halfwidth = (p[kind + '_hat'], p['epsilon_' + kind])
    intervals = [(center - halfwidth, center + halfwidth)]
    goal = cfg.target_B_error if kind == 'B' else cfg.target_C_error
    history = []
    status = 'target_reached' if halfwidth <= goal else 'time_or_frequency_limit'
    for m in periods:
        if interval_estimate(intervals)[1] <= goal:
            status = 'target_reached'
            break
        t = 2 * np.pi * m / p['omega_hat']
        measured = simulate_ramsey_experiment(label, kind, m, p['omega_hat'], cfg.shots, rng, source)
        phase, measurement_error = phase_from_measurement(measured, cfg.shots, per_pair_failure)
        model_error = systematic_phase_bound(kind, t, p)
        gain = t if kind == 'B' else 2 * t / np.pi
        updated = intersect_phase_candidates(intervals, phase, gain, model_error + measurement_error + cfg.extra_phase_error, cfg.max_alias_candidates)
        if not updated:
            raise RuntimeError(f'{label}/{kind}: 第 m={m} 周期的区间交集为空。检查上游区间、噪声、控制误差或模型，不能强行输出高精度结果。')
        intervals = updated
        estimate, radius = interval_estimate(intervals)
        history.append(dict(m=m, t=t, theta_mod=phase, measurement=measured, measurement_phase_error=measurement_error, model_phase_error_bound=model_error, estimate=estimate, error_bound=radius, intervals=intervals.copy()))
    estimate, radius = interval_estimate(intervals)
    if radius <= goal:
        status = 'target_reached'
    elif not periods:
        status = 'no_admissible_period'
    elif len(intervals) > 1:
        status = 'unresolved_phase_aliases'
    return dict(estimate=estimate, error_bound=radius, intervals=intervals, status=status, history=history, target_error=goal, max_interrogation_time=max((x['t'] for x in history), default=0.0), sum_interrogation_times=sum((x['t'] for x in history)), total_sensing_time=None if cfg.shots is None else 2 * cfg.shots * sum((x['t'] for x in history)))

def refine_all_targets(frequency_results: Optional[Mapping]=None, config: Optional[Config]=None, source=None) -> dict:
    source = DEFAULT_ORACLE if source is None else source
    cfg = Config() if config is None else config
    validate_shots(cfg.shots)
    if not np.isfinite([cfg.target_B_error, cfg.target_C_error, cfg.max_B_time, cfg.max_C_time, cfg.max_C_model_phase_error, cfg.extra_phase_error]).all():
        raise ValueError('Config 数值必须有限。')
    if not 0 < cfg.measurement_failure_probability < 1:
        raise ValueError('测量失败概率必须在 (0,1)。')
    if min(cfg.target_B_error, cfg.target_C_error, cfg.max_B_time, cfg.max_C_time, cfg.max_C_model_phase_error) <= 0 or cfg.extra_phase_error < 0:
        raise ValueError('目标误差/时间/模型预算必须为正，额外相位误差非负。')
    if frequency_results is None:
        frequency_results = learn_omega_for_all_targets(source=source)
    inputs = frequency_results
    params = {lab: normalize_input(rec) for lab, rec in inputs.items()}
    schedules = {(lab, kind): synchronized_schedule(kind, p, cfg) for lab, p in params.items() for kind in ('B', 'C')}
    max_pairs = sum((len(x) for x in schedules.values()))
    per_pair_failure = cfg.measurement_failure_probability / max(1, max_pairs)
    results = {}
    for index, (label, p) in enumerate(params.items()):
        rng = np.random.default_rng(cfg.seed + 1000 * index)
        b = refine_one_coefficient(label, 'B', p, cfg, schedules[label, 'B'], per_pair_failure, rng, source)
        c = refine_one_coefficient(label, 'C', p, cfg, schedules[label, 'C'], per_pair_failure, rng, source)
        result = dict(target_label=label, inputs=p, B=b, C=c, B_hat_ref=b['estimate'], C_hat_ref=c['estimate'], epsilon_B_ref=b['error_bound'], epsilon_C_ref=c['error_bound'], refinement_measurement_failure_probability=0.0 if cfg.shots is None else cfg.measurement_failure_probability, upstream_search_failure_probability_bound=inputs[label].get('search_failure_probability_bound'), guarantee='conditional on upstream intervals and ideal reshaping/control')
        if label in source.by_label:
            truth = source.by_label[label]
            result['diagnostics'] = dict(B_true=truth.B, C_true=truth.C, omega_true=truth.omega, B_error_before=abs(p['B_hat'] - truth.B), C_error_before=abs(p['C_hat'] - truth.C), B_error_after=abs(b['estimate'] - truth.B), C_error_after=abs(c['estimate'] - truth.C), initial_intervals_valid=abs(p['B_hat'] - truth.B) <= p['epsilon_B'] and abs(p['C_hat'] - truth.C) <= p['epsilon_C'] and (p['omega_lower'] <= truth.omega <= p['omega_upper']), final_intervals_cover_truth=any((lo <= truth.B <= hi for lo, hi in b['intervals'])) and any((lo <= truth.C <= hi for lo, hi in c['intervals'])))
        results[label] = result
    return results

def print_results(results):
    print('Pauli        B_ref           B_bound           C_ref           C_bound')
    for label, r in results.items():
        print(f"{label:4s} {r['B_hat_ref']:16.10f} {r['epsilon_B_ref']:15.6e} {r['C_hat_ref']:16.10f} {r['epsilon_C_ref']:15.6e}")
        print('  B:', r['B']['status'], 'C:', r['C']['status'])


# ===== scan_lambda.py =====
import argparse
import csv
import dataclasses
import json
from pathlib import Path
import numpy as np

def serialize(x):
    if dataclasses.is_dataclass(x):
        return dataclasses.asdict(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)

def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys((k for row in rows for k in row)))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def make_plots(summary, preliminary, target, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    good = sorted((r for r in summary if r['successful_trials'] > 0), key=lambda r: r['lambda'])
    if not good:
        return
    x = np.array([r['lambda'] for r in good])
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, kind, label in zip(axes, ('omega', 'B', 'C'), ('Frequency', 'Static coefficient B', 'Modulation amplitude C')):
        mean = np.array([r[kind + '_error_mean'] for r in good])
        p10 = np.array([r[kind + '_error_p10'] for r in good])
        p90 = np.array([r[kind + '_error_p90'] for r in good])
        bound = np.array([r[kind + '_bound_mean'] for r in good])
        floor = 1e-16
        ax.plot(x, np.maximum(mean, floor), 'o-', color='#1764ab', label='Mean actual error')
        ax.fill_between(x, np.maximum(p10, floor), np.maximum(p90, floor), color='#1764ab', alpha=0.16, label='10–90% of runs')
        ax.plot(x, np.maximum(bound, floor), 's--', color='#ce571d', label='Mean reported bound')
        if kind in ('B', 'C'):
            ax.axhline(max(preliminary['epsilon_' + kind], floor), color='.4', ls=':', label='Preliminary bound')
        ax.set(xlabel='Attenuation $\\lambda$', ylabel='Absolute error / bound', title=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f'{target}: lambda sweep; fixed preliminary estimates and refinement budgets', fontsize=13)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'lambda_accuracy_{target}.{extension}', dpi=220)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    for ax, key, label in zip(axes, ('T_ESU', 'K', 'ESU_queries_mean'), ('One ESU interrogation time', 'Candidate count K', 'Mean ESU queries')):
        ax.plot(x, [r[key] for r in good], 'o-', color='#1764ab')
        ax.set(xlabel='Attenuation $\\lambda$', ylabel=label)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which='both', alpha=0.2)
    fig.suptitle(f'{target}: resource cost', fontsize=13)
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'lambda_resources_{target}.{extension}', dpi=220)
    plt.close(fig)

def scan(target=TARGET_LABEL, lambdas=None, repeats=REPEATS, shots=None, T_B=128.0, t_short=0.2, max_B_time=1024.0, max_C_time=256.0, target_B_error=1e-06, target_C_error=1e-06, failure_probability=0.01, seed=42, maximum_bins=4096, output='lambda_three_results', verbose=True, source=None):
    lambdas = LAMBDA_VALUES if lambdas is None else list(lambdas)
    if not lambdas or not np.isfinite(lambdas).all() or any((not 0 < v <= 1 for v in lambdas)):
        raise ValueError('Each lambda must be finite and in (0,1].')
    if len(set(lambdas)) != len(lambdas):
        raise ValueError('Duplicate lambda values.')
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats <= 0:
        raise ValueError('repeats must be a positive integer.')
    if not 0 < failure_probability <= 0.5:
        raise ValueError('Invalid per-run search failure budget.')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = ThreeQubitOracle() if source is None else source
    structure = learn_structure(source=source)
    support = structure['support']
    if target not in support:
        raise ValueError(f'{target} was not identified in {support}.')
    priors = EstimationPriors()
    preliminary = learn_B_and_C_for_target(target, support, source, priors, T_B=T_B, t_short=t_short, shots=shots, failure_probability=0.02, seed=123)
    trials = []
    summary = []
    design_records = []
    for value in lambdas:
        if verbose:
            print(f'\nlambda = {value:g}', flush=True)
        info = dict(lambda_value=value)
        try:
            d = choose_search_design(preliminary, priors.omega_min, priors.omega_max, failure_probability=failure_probability, maximum_bins=maximum_bins, fixed_scale=value)
            if d.scale != value:
                raise RuntimeError('The requested lambda was changed.')
            bins = build_frequency_grid(d)
            phases, certificate = synthesize_qsp_phases(d, verbose=verbose)
            theta = np.array([source.esu_phase(target, w, d.T_total, d.B_hat, d.scale) for w in bins])
            blocks = build_qsp_oracle_blocks(theta, phases)
            info.update(status='ready', design=dataclasses.asdict(d), qsp_phases=phases, qsp_degree=len(phases) - 1, filter_certificate=certificate)
        except (ValueError, RuntimeError) as exc:
            info.update(status='design_failed', error=str(exc))
            summary.append({'lambda': value, 'status': 'design_failed', 'successful_trials': 0, 'failed_trials': 0, 'skipped_trials': repeats, 'error': str(exc)})
            design_records.append(info)
            if verbose:
                print('  DESIGN FAILED:', exc, flush=True)
            continue
        design_records.append(info)
        batch = []
        for trial in range(repeats):
            row = dict(target=target, **{'lambda': value}, trial=trial, status='ok', T_ESU=d.T_total, K=d.number_of_bins, correction_bound=d.correction_bound, qsp_degree=len(phases) - 1, qsp_error_bound=certificate['continuous_error_bound'])
            try:
                found = bbht_amplitude_search(blocks, len(phases) - 1, seed=seed + trial)
                record = dict(target_label=target, design=d, omega_hat=float(bins[found['k_hat']]), frequency_error_bound=d.r_out / d.T_total, search_failure_probability_bound=failure_probability)
                cfg = Config(shots=shots, max_B_time=max_B_time, max_C_time=max_C_time, target_B_error=target_B_error, target_C_error=target_C_error, measurement_failure_probability=0.01, seed=20261012 + trial)
                refined = refine_all_targets({target: record}, cfg, source=source)[target]
                row.update(omega_hat=record['omega_hat'], omega_bound=record['frequency_error_bound'], ESU_queries=found['esu_queries'], frequency_sensing_time=found['esu_queries'] * d.T_total, bbht_rounds=found['bbht_rounds'], B_hat_ref=refined['B_hat_ref'], B_bound=refined['epsilon_B_ref'], C_hat_ref=refined['C_hat_ref'], C_bound=refined['epsilon_C_ref'], B_status=refined['B']['status'], C_status=refined['C']['status'], B_max_time=refined['B']['max_interrogation_time'], C_max_time=refined['C']['max_interrogation_time'], refinement_sensing_time=None if shots is None else refined['B']['total_sensing_time'] + refined['C']['total_sensing_time'])
                truth = source.by_label[target]
                row.update(omega_true=truth.omega, B_true=truth.B, C_true=truth.C, omega_error=abs(row['omega_hat'] - truth.omega), B_error=abs(row['B_hat_ref'] - truth.B), C_error=abs(row['C_hat_ref'] - truth.C))
                for kind in ('omega', 'B', 'C'):
                    row[kind + '_covered'] = row[kind + '_error'] <= row[kind + '_bound'] + 1e-12
                batch.append(row)
            except (ValueError, RuntimeError) as exc:
                row.update(status='trial_failed', error=str(exc))
            trials.append(row)
        s = dict(**{'lambda': value}, status='ok' if len(batch) == repeats else 'partial_or_failed', successful_trials=len(batch), failed_trials=repeats - len(batch), skipped_trials=0, T_ESU=d.T_total, K=d.number_of_bins, correction_bound=d.correction_bound, qsp_degree=len(phases) - 1, qsp_error_bound=certificate['continuous_error_bound'])
        if batch:
            for kind in ('omega', 'B', 'C'):
                errors = np.array([r[kind + '_error'] for r in batch])
                s.update({kind + '_error_mean': float(errors.mean()), kind + '_error_median': float(np.median(errors)), kind + '_error_p10': float(np.quantile(errors, 0.1)), kind + '_error_p90': float(np.quantile(errors, 0.9)), kind + '_bound_mean': float(np.mean([r[kind + '_bound'] for r in batch])), kind + '_coverage_fraction': float(np.mean([r[kind + '_covered'] for r in batch]))})
            s['ESU_queries_mean'] = float(np.mean([r['ESU_queries'] for r in batch]))
            s['frequency_sensing_time_mean'] = float(np.mean([r['frequency_sensing_time'] for r in batch]))
            if verbose:
                print(f"  success={len(batch)}/{repeats}, T={d.T_total:.3f}, K={d.number_of_bins}; omega mean error={s['omega_error_mean']:.3e}, bound={s['omega_bound_mean']:.3e}; B bound={s['B_bound_mean']:.3e}, C bound={s['C_bound_mean']:.3e}", flush=True)
        summary.append(s)
        write_csv(output / 'trials.csv', trials)
        write_csv(output / 'summary.csv', summary)
    write_csv(output / 'trials.csv', trials)
    write_csv(output / 'summary.csv', summary)
    meta = dict(target=target, lambdas=lambdas, repeats=repeats, shots=shots, seed=seed, preliminary=preliminary, designs=design_records, summary=summary, fixed_refinement_settings=dict(max_B_time=max_B_time, max_C_time=max_C_time, target_B_error=target_B_error, target_C_error=target_C_error), per_run_search_failure_probability=failure_probability, all_searches_union_failure_bound=min(1.0, len(lambdas) * repeats * failure_probability), note='Shared preliminary uncertainty is not independent across trials. Means/quantiles use successful trials only; failures are counted separately. Individual-run bounds are not simultaneous confidence intervals for all rows.', assumptions='Ideal reshaping/attenuation/coherent ESU; floating-point QSP certificate. None shots is infinite-shot expectation.')
    (output / 'details.json').write_text(json.dumps(meta, default=serialize, ensure_ascii=False, indent=2), encoding='utf-8')
    make_plots(summary, preliminary, target, output)
    return (meta, trials)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', default=TARGET_LABEL, choices=['XII', 'IXX', 'ZXY'])
    parser.add_argument('--lambdas', type=float, nargs='+', default=LAMBDA_VALUES)
    parser.add_argument('--repeats', type=int, default=REPEATS)
    parser.add_argument('--shots', type=int, default=None)
    parser.add_argument('--T-B', type=float, default=128.0)
    parser.add_argument('--t-short', type=float, default=0.2)
    parser.add_argument('--max-B-time', type=float, default=1024.0)
    parser.add_argument('--max-C-time', type=float, default=256.0)
    parser.add_argument('--target-B-error', type=float, default=1e-06)
    parser.add_argument('--target-C-error', type=float, default=1e-06)
    parser.add_argument('--failure-probability', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--maximum-bins', type=int, default=4096)
    parser.add_argument('--output', default='lambda_three_results')
    args = vars(parser.parse_args())
    result, rows = scan(**args)
    print('\nSaved results to', Path(args['output']).resolve())
