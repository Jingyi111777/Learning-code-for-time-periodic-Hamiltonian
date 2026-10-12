import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la


# ============================================================
# 1. Basic Pauli matrices
# ============================================================
I2 = np.eye(2, dtype=complex)

X = np.array([
    [0, 1],
    [1, 0],
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0],
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1],
], dtype=complex)

SINGLE_QUBIT_PAULI = {
    "I": I2,
    "X": X,
    "Y": Y,
    "Z": Z,
}


# ============================================================
# 2. Three-qubit Pauli strings
# ============================================================
def pauli_string_matrix(label: str) -> np.ndarray:
    if len(label) != 3:
        raise ValueError("A three-qubit Pauli label must have length 3.")

    if any(ch not in SINGLE_QUBIT_PAULI for ch in label):
        raise ValueError(f"Invalid Pauli label: {label}")

    return np.kron(
        SINGLE_QUBIT_PAULI[label[2]],
        np.kron(
            SINGLE_QUBIT_PAULI[label[1]],
            SINGLE_QUBIT_PAULI[label[0]],
        ),
    )


ALL_PAULI_LABELS = [
    "".join(chars)
    for chars in itertools.product("IXYZ", repeat=3)
]

ALL_PAULI_MATRICES = {
    label: pauli_string_matrix(label)
    for label in ALL_PAULI_LABELS
}

DIM_SYSTEM = 2**3
I8 = np.eye(DIM_SYSTEM, dtype=complex)


# ============================================================
# 3. Three-qubit periodically time-dependent Hamiltonian
#
# H(t) =
# [0.72 + 0.08 cos(1.0 t)] XII
# + [0.59 + 0.06 cos(1.1 t)] ZZI
# + [0.36 + 0.04 cos(1.2 t)] IXX
# + [0.29 + 0.03 cos(1.3 t)] YIZ
# + [0.18 + 0.02 cos(1.4 t)] XYZ
# + [0.145+0.015 cos(1.5 t)] ZXY
# ============================================================
@dataclass(frozen=True)
class PeriodicPauliTerm:
    label: str
    B: float
    C: float
    omega: float

    def coefficient(self, t: float) -> float:
        return self.B + self.C * np.cos(self.omega * t)


TERMS: Tuple[PeriodicPauliTerm, ...] = (
    PeriodicPauliTerm("XII", 0.72,  0.08,  1.0),
    PeriodicPauliTerm("ZZI", 0.59,  0.06,  1.1),

    PeriodicPauliTerm("IXX", 0.36,  0.04,  1.2),
    PeriodicPauliTerm("YIZ", 0.29,  0.03,  1.3),

    PeriodicPauliTerm("XYZ", 0.18,  0.02,  1.4),
    PeriodicPauliTerm("ZXY", 0.145, 0.015, 1.5),
)

TERM_BY_LABEL = {
    term.label: term
    for term in TERMS
}

LEARNED_SUPPORT = [
    term.label
    for term in TERMS
]


def H_three_qubit(t: float) -> np.ndarray:
    """Full three-qubit unknown Hamiltonian H(t)."""
    H_mat = np.zeros((DIM_SYSTEM, DIM_SYSTEM), dtype=complex)

    for term in TERMS:
        H_mat += (
            term.coefficient(t)
            * ALL_PAULI_MATRICES[term.label]
        )

    return H_mat


# ============================================================
# 4.1 Pauli commutation analysis in symplectic space
# ============================================================
PAULI_TO_SYMPLECTIC_BITS = {
    "I": (0, 0),
    "X": (1, 0),
    "Z": (0, 1),
    "Y": (1, 1),
}
def pauli_to_symplectic(
    label: str,
) -> np.ndarray:
    
    if len(label) != 3:
        raise ValueError(
            "A three-qubit Pauli label must have length 3."
        )

    x_bits = []
    z_bits = []

    for ch in label:
        if ch not in PAULI_TO_SYMPLECTIC_BITS:
            raise ValueError(
                f"Invalid Pauli label: {label}"
            )

        x, z = PAULI_TO_SYMPLECTIC_BITS[ch]
        x_bits.append(x)
        z_bits.append(z)

    return np.asarray(
        x_bits + z_bits,
        dtype=np.uint8,
    )


def symplectic_to_pauli(
    vector: Sequence[int],
) -> str:
   
    v = np.asarray(
        vector,
        dtype=np.uint8,
    ) % 2

    if v.shape != (6,):
        raise ValueError(
            "A three-qubit symplectic vector must have length 6."
        )

    x = v[:3]
    z = v[3:]

    label_chars = []

    for x_j, z_j in zip(x, z):
        pair = (
            int(x_j),
            int(z_j),
        )

        if pair == (0, 0):
            label_chars.append("I")
        elif pair == (1, 0):
            label_chars.append("X")
        elif pair == (0, 1):
            label_chars.append("Z")
        elif pair == (1, 1):
            label_chars.append("Y")

    return "".join(label_chars)


def symplectic_inner(
    u: Sequence[int],
    v: Sequence[int],
) -> int:
    
    u = np.asarray(
        u,
        dtype=np.uint8,
    ) % 2

    v = np.asarray(
        v,
        dtype=np.uint8,
    ) % 2

    if u.shape != (6,) or v.shape != (6,):
        raise ValueError(
            "Three-qubit symplectic vectors must have length 6."
        )

    x_u = u[:3]
    z_u = u[3:]

    x_v = v[:3]
    z_v = v[3:]

    return int(
        (
            np.dot(x_u, z_v)
            + np.dot(z_u, x_v)
        )
        % 2
    )


def pauli_commutes(
    label_a: str,
    label_b: str,
) -> bool:
    
    u = pauli_to_symplectic(
        label_a
    )
    v = pauli_to_symplectic(
        label_b
    )

    return (
        symplectic_inner(u, v)
        == 0
    )

# ============================================================
#4.2. Automatic Hamiltonian-reshaping sequence construction
# ============================================================

def pauli_weight(
    label: str,
) -> int:
    
    return sum(
        ch != "I"
        for ch in label
    )


def gf2_rank(
    vectors: Sequence[Sequence[int]],
) -> int:
    if len(vectors) == 0:
        return 0

    matrix = np.asarray(
        vectors,
        dtype=np.uint8,
    ) % 2

    matrix = matrix.copy()

    n_rows, n_cols = matrix.shape

    rank = 0
    pivot_col = 0

    while (
        rank < n_rows
        and pivot_col < n_cols
    ):
        pivot_rows = np.where(
            matrix[rank:, pivot_col] == 1
        )[0]

        if len(pivot_rows) == 0:
            pivot_col += 1
            continue

        pivot_row = (
            rank
            + int(pivot_rows[0])
        )

        if pivot_row != rank:
            matrix[
                [rank, pivot_row]
            ] = matrix[
                [pivot_row, rank]
            ]

        for row in range(n_rows):
            if (
                row != rank
                and matrix[row, pivot_col] == 1
            ):
                matrix[row] ^= matrix[rank]

        rank += 1
        pivot_col += 1

    return int(rank)


def generators_are_independent(
    generator_labels: Sequence[str],
) -> bool:
   
    vectors = [
        pauli_to_symplectic(label)
        for label in generator_labels
    ]

    return (
        gf2_rank(vectors)
        == len(generator_labels)
    )


def find_reshaping_generators(
    target_label: str,
    support: Sequence[str] = LEARNED_SUPPORT,
) -> Tuple[str, ...]:
    
    if target_label == "III":
        raise ValueError(
            "The target Pauli cannot be III."
        )

    if target_label not in support:
        raise ValueError(
            f"Target {target_label} is not in the learned support."
        )

    target_vector = pauli_to_symplectic(
        target_label
    )

    non_targets = [
        label
        for label in support
        if label != target_label
    ]

    if len(non_targets) == 0:
        return tuple()

    # Candidate generators must preserve the target:
    # omega(g, target) = 0.
    candidates = []

    for label in ALL_PAULI_LABELS:
        if label == "III":
            continue

        g_vector = pauli_to_symplectic(
            label
        )

        if (
            symplectic_inner(
                g_vector,
                target_vector,
            )
            == 0
        ):
            candidates.append(label)

    # Prefer experimentally simpler, lower-weight generators.
    candidates.sort(
        key=lambda label: (
            pauli_weight(label),
            label,
        )
    )

    # Search for the smallest mutually commuting,
    # linearly independent generator set that cancels
    # every non-target term.
    max_generators = min(
        len(non_targets),
        6,
    )

    for number_of_generators in range(
        1,
        max_generators + 1,
    ):
        for generators in itertools.combinations(
            candidates,
            number_of_generators,
        ):
            # The generators should form an Abelian subgroup.
            pairwise_commuting = all(
                pauli_commutes(g1, g2)
                for g1, g2
                in itertools.combinations(
                    generators,
                    2,
                )
            )

            if not pairwise_commuting:
                continue

            # Independent generators give exactly 2^r
            # distinct symplectic group elements.
            if not generators_are_independent(
                generators
            ):
                continue

            # Every non-target term must acquire a minus sign
            # under at least one generator.
            all_non_targets_cancelled = True

            for other_label in non_targets:
                other_vector = pauli_to_symplectic(
                    other_label
                )

                anticommutes_with_some_generator = any(
                    symplectic_inner(
                        pauli_to_symplectic(g),
                        other_vector,
                    )
                    == 1
                    for g in generators
                )

                if not anticommutes_with_some_generator:
                    all_non_targets_cancelled = False
                    break

            if all_non_targets_cancelled:
                return tuple(generators)

    raise RuntimeError(
        "No valid symplectic reshaping-generator set "
        f"was found for target {target_label}."
    )


def build_reshaping_sequence_labels(
    target_label: str,
) -> Tuple[List[str], Tuple[str, ...]]:
    
    generators = find_reshaping_generators(
        target_label=target_label,
        support=LEARNED_SUPPORT,
    )

    sequence_vectors = [
        np.zeros(
            6,
            dtype=np.uint8,
        )
    ]

    for generator_label in generators:
        g_vector = pauli_to_symplectic(
            generator_label
        )

        old_vectors = [
            vector.copy()
            for vector in sequence_vectors
        ]

        sequence_vectors.extend([
            old_vector ^ g_vector
            for old_vector in old_vectors
        ])

    sequence_labels = [
        symplectic_to_pauli(vector)
        for vector in sequence_vectors
    ]

    return (
        sequence_labels,
        generators,
    )


def build_reshaping_sequence(
    target_label: str,
) -> List[np.ndarray]:
    
    sequence_labels, _ = (
        build_reshaping_sequence_labels(
            target_label
        )
    )

    return [
        ALL_PAULI_MATRICES[label]
        for label in sequence_labels
    ]


def verify_reshaping_sequence(
    target_label: str,
    atol: float = 1e-10,
) -> bool:
    
    sequence = build_reshaping_sequence(
        target_label
    )

    L = len(sequence)

    P_target = ALL_PAULI_MATRICES[
        target_label
    ]

    target_average = sum(
        Q.conj().T @ P_target @ Q
        for Q in sequence
    ) / L

    if not np.allclose(
        target_average,
        P_target,
        atol=atol,
    ):
        return False

    for label in LEARNED_SUPPORT:
        if label == target_label:
            continue

        P_other = ALL_PAULI_MATRICES[
            label
        ]

        other_average = sum(
            Q.conj().T @ P_other @ Q
            for Q in sequence
        ) / L

        if not np.allclose(
            other_average,
            np.zeros_like(P_other),
            atol=atol,
        ):
            return False

    return True
# ============================================================
# 5. Reshaped three-qubit evolution without ESU
# ============================================================
def apply_one_reshaping_cycle(
    state: np.ndarray,
    target_label: str,
    t_left: float,
    cycle_dt: float,
    sequence: Sequence[np.ndarray],
) -> np.ndarray:
    out = np.asarray(
        state,
        dtype=complex,
    ).reshape(DIM_SYSTEM)

    L = len(sequence)
    sub_dt = cycle_dt / L

    for m, Q in enumerate(sequence):
        t_mid = t_left + (m + 0.5) * sub_dt

        H_switch = (
            Q.conj().T
            @ H_three_qubit(t_mid)
            @ Q
        )

        U_step = la.expm(
            -1j * H_switch * sub_dt
        )

        out = U_step @ out

    return out


def apply_reshaped_no_esu_state(
    initial_state: np.ndarray,
    target_label: str,
    T_total: float,
    cycle_dt_max: float = 0.01,
) -> np.ndarray:
    """
    Apply the complete reshaped evolution from t=0 to T_total.

    """
    if T_total < 0:
        raise ValueError("T_total must be nonnegative.")

    if cycle_dt_max <= 0:
        raise ValueError("cycle_dt_max must be positive.")

    sequence = build_reshaping_sequence(target_label)

    out = np.asarray(
        initial_state,
        dtype=complex,
    ).reshape(DIM_SYSTEM)

    if T_total == 0:
        return out.copy()

    n_cycles = max(
        1,
        int(np.ceil(T_total / cycle_dt_max)),
    )

    cycle_dt = T_total / n_cycles
    t_left = 0.0

    for _ in range(n_cycles):
        out = apply_one_reshaping_cycle(
            state=out,
            target_label=target_label,
            t_left=t_left,
            cycle_dt=cycle_dt,
            sequence=sequence,
        )

        t_left += cycle_dt

    return out


# ============================================================
# 6.Generalized Ramsey probe and observable construction
# ============================================================
def choose_reference_pauli(
    target_label: str,
) -> str:
    
    chars = ["I", "I", "I"]

    for i, ch in enumerate(target_label):
        if ch == "I":
            continue

        if ch in ("X", "Y"):
            chars[i] = "Z"
        elif ch == "Z":
            chars[i] = "X"

        return "".join(chars)

    raise ValueError("The target Pauli cannot be III.")


def plus_eigenstate_single_qubit(
    pauli_char: str,
) -> np.ndarray:
    
    if pauli_char in ("I", "Z"):
        return np.array(
            [1.0, 0.0],
            dtype=complex,
        )

    if pauli_char == "X":
        return (
            np.array(
                [1.0, 1.0],
                dtype=complex,
            )
            / np.sqrt(2.0)
        )

    if pauli_char == "Y":
        return (
            np.array(
                [1.0, 1.0j],
                dtype=complex,
            )
            / np.sqrt(2.0)
        )

    raise ValueError(
        f"Invalid single-qubit Pauli character: {pauli_char}"
    )


def prepare_reference_plus_state(
    reference_label: str,
) -> np.ndarray:
    
    q0 = plus_eigenstate_single_qubit(
        reference_label[0]
    )
    q1 = plus_eigenstate_single_qubit(
        reference_label[1]
    )
    q2 = plus_eigenstate_single_qubit(
        reference_label[2]
    )

    return np.kron(
        q2,
        np.kron(q1, q0),
    )


def identify_signed_pauli_matrix(
    matrix: np.ndarray,
    atol: float = 1e-10,
) -> Tuple[int, str]:
    
    for label, P in ALL_PAULI_MATRICES.items():
        if np.allclose(
            matrix,
            P,
            atol=atol,
        ):
            return +1, label

        if np.allclose(
            matrix,
            -P,
            atol=atol,
        ):
            return -1, label

    raise ValueError(
        "The matrix is not a signed three-qubit Pauli string."
    )


def ramsey_observables(
    target_label: str,
) -> Dict[str, object]:
    P = ALL_PAULI_MATRICES[target_label]

    reference_label = choose_reference_pauli(
        target_label
    )
    R = ALL_PAULI_MATRICES[reference_label]

    # S = -i P R is Hermitian because {P,R}=0.
    S_matrix = -1j * P @ R

    S_sign, S_label = identify_signed_pauli_matrix(
        S_matrix
    )

    return {
        "reference_label": reference_label,
        "R_matrix": R,
        "S_matrix": S_matrix,
        "S_sign": S_sign,
        "S_label": S_label,
    }


# ============================================================
# 7. Pauli measurement
# ============================================================
def measure_pauli_expectation(
    state: np.ndarray,
    observable: np.ndarray,
    shots: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, object]:
    
    psi = np.asarray(
        state,
        dtype=complex,
    ).reshape(DIM_SYSTEM)

    exp_exact = float(
        np.real(
            np.vdot(
                psi,
                observable @ psi,
            )
        )
    )

    # Remove tiny numerical overshoots outside [-1,1].
    exp_exact = float(
        np.clip(
            exp_exact,
            -1.0,
            1.0,
        )
    )

    p_plus = 0.5 * (
        1.0 + exp_exact
    )
    p_minus = 1.0 - p_plus

    if shots is None:
        return {
            "exp_val": exp_exact,
            "counts": None,
            "probs": np.array(
                [p_plus, p_minus],
                dtype=float,
            ),
        }

    rng = np.random.default_rng(seed)

    n_plus = int(
        rng.binomial(
            shots,
            p_plus,
        )
    )
    n_minus = shots - n_plus

    exp_sampled = (
        n_plus - n_minus
    ) / shots

    return {
        "exp_val": float(exp_sampled),
        "counts": np.array(
            [n_plus, n_minus],
            dtype=int,
        ),
        "probs": np.array(
            [p_plus, p_minus],
            dtype=float,
        ),
    }


# ============================================================
# 8. Ramsey trajectory
# ============================================================
def ramsey_trajectory_for_target(
    target_label: str,
    times: Sequence[float],
    cycle_dt_max: float = 0.01,
    shots: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, object]:

    times = np.asarray(
        times,
        dtype=float,
    )

    if len(times) == 0:
        raise ValueError("times cannot be empty.")

    if np.any(times <= 0):
        raise ValueError(
            "All Ramsey sampling times must be positive."
        )

    if len(times) > 1:
        sample_dt = times[0]

        if not np.allclose(
            times,
            sample_dt
            * np.arange(
                1,
                len(times) + 1,
            ),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "This optimized trajectory function expects "
                "equally spaced times t_j = j * sample_dt."
            )
    else:
        sample_dt = times[0]

    
    cycles_per_sample = max(
        1,
        int(
            np.ceil(
                sample_dt / cycle_dt_max
            )
        ),
    )

    cycle_dt = (
        sample_dt
        / cycles_per_sample
    )

    sequence = build_reshaping_sequence(
        target_label
    )

    obs = ramsey_observables(
        target_label
    )

    state = prepare_reference_plus_state(
        obs["reference_label"]
    )

    R_exps = []
    S_exps = []
    theta_mods = []

    R_counts = []
    S_counts = []

    R_probs = []
    S_probs = []

    current_time = 0.0

    for j in range(len(times)):
        # Propagate from t_j to t_{j+1}.
        for _ in range(cycles_per_sample):
            state = apply_one_reshaping_cycle(
                state=state,
                target_label=target_label,
                t_left=current_time,
                cycle_dt=cycle_dt,
                sequence=sequence,
            )

            current_time += cycle_dt

        r_measure = measure_pauli_expectation(
            state=state,
            observable=obs["R_matrix"],
            shots=shots,
            seed=seed + 2 * j,
        )

        s_measure = measure_pauli_expectation(
            state=state,
            observable=obs["S_matrix"],
            shots=shots,
            seed=seed + 2 * j + 1,
        )

        r_exp = float(
            r_measure["exp_val"]
        )
        s_exp = float(
            s_measure["exp_val"]
        )

        theta_mod = (
            0.5
            * np.arctan2(
                s_exp,
                r_exp,
            )
        )

        R_exps.append(r_exp)
        S_exps.append(s_exp)
        theta_mods.append(
            float(theta_mod)
        )

        R_counts.append(
            r_measure["counts"]
        )
        S_counts.append(
            s_measure["counts"]
        )

        R_probs.append(
            r_measure["probs"]
        )
        S_probs.append(
            s_measure["probs"]
        )

    return {
        "target_label": target_label,
        "reference_label":
            obs["reference_label"],
        "second_observable_sign":
            obs["S_sign"],
        "second_observable_label":
            obs["S_label"],
        "times": times,
        "R_exps": np.asarray(
            R_exps,
            dtype=float,
        ),
        "S_exps": np.asarray(
            S_exps,
            dtype=float,
        ),
        "theta_mods": np.asarray(
            theta_mods,
            dtype=float,
        ),
        "R_counts": np.asarray(
            R_counts,
            dtype=object,
        ),
        "S_counts": np.asarray(
            S_counts,
            dtype=object,
        ),
        "R_probs": np.asarray(
            R_probs,
            dtype=float,
        ),
        "S_probs": np.asarray(
            S_probs,
            dtype=float,
        ),
        "cycle_dt": float(
            cycle_dt
        ),
        "cycles_per_sample":
            int(cycles_per_sample),
        "reshaping_sequence_length":
            len(sequence),
    }


# ============================================================
# 9. Unwrap theta measured modulo pi
# ============================================================
def unwrap_half_angles(
    theta_mods: Sequence[float],
) -> np.ndarray:
    theta_mods = np.asarray(
        theta_mods,
        dtype=float,
    )

    return (
        0.5
        * np.unwrap(
            2.0 * theta_mods
        )
    )


# ============================================================
# 10. Short-time rough estimate of B + C

# ============================================================
def estimate_B_plus_C_short_time(
    target_label: str,
    t_short: float = 0.03,
    cycle_dt_max: float = 0.003,
    C_max: float = 0.1,
    omega_max: float = 1.5,
    shots: Optional[int] = None,
    seed: int = 123,
) -> Dict[str, object]:

    result = ramsey_trajectory_for_target(
        target_label=target_label,
        times=[t_short],
        cycle_dt_max=cycle_dt_max,
        shots=shots,
        seed=seed,
    )

    theta_short = float(
        result["theta_mods"][0]
    )

    B_plus_C_hat = (
        theta_short
        / t_short
    )

    epsilon_S = (
        C_max
        * omega_max**2
        * t_short**2
        / 6.0
    )

    return {
        "target_label": target_label,
        "B_plus_C_hat":
            float(B_plus_C_hat),
        "theta_short":
            float(theta_short),
        "epsilon_S":
            float(epsilon_S),
        "ramsey_result":
            result,
    }


# ============================================================
# 11. Long-time fitting for B and estimation of C

# ============================================================
def learn_B_and_C_for_target(
    target_label: str,
    T_total: float = 30.0,
    t_short: float = 0.03,
    short_cycle_dt_max: float = 0.003,
    long_cycle_dt_max: float = 0.01,
    safety_factor: float = 0.2,
    B_max: float = 0.8,
    C_max: float = 0.1,
    omega_min: float = 1.0,
    omega_max: float = 1.5,
    shots: Optional[int] = None,
    seed: int = 1000,
) -> Dict[str, object]:

    if target_label not in TERM_BY_LABEL:
        raise ValueError(
            f"Unknown target label: {target_label}"
        )

    if B_max + C_max <= 0:
        raise ValueError(
            "B_max + C_max must be positive."
        )

    if omega_min <= 0:
        raise ValueError(
            "omega_min must be positive."
        )

    # --------------------------------------------------------
    # Step 1: short-time estimate of S = B + C
    # --------------------------------------------------------
    short_result = (
        estimate_B_plus_C_short_time(
            target_label=target_label,
            t_short=t_short,
            cycle_dt_max=
                short_cycle_dt_max,
            C_max=C_max,
            omega_max=omega_max,
            shots=shots,
            seed=seed,
        )
    )

    B_plus_C_hat = float(
        short_result["B_plus_C_hat"]
    )

    epsilon_S = float(
        short_result["epsilon_S"]
    )

    # --------------------------------------------------------
    # Step 2: choose Ramsey sampling interval
    # --------------------------------------------------------
    sample_dt = (
        safety_factor
        * np.pi
        / (
            2.0
            * (B_max + C_max)
        )
    )

    times = np.arange(
        sample_dt,
        T_total + 1e-12,
        sample_dt,
    )

    N = len(times)

    if N == 0:
        raise ValueError(
            "T_total is too small for the chosen sample_dt."
        )

    epsilon_B = (
        3.0
        * C_max
        / (
            omega_min
            * sample_dt
            * (2 * N + 1)
        )
    )

    # --------------------------------------------------------
    # Step 3: Ramsey measurements at t_j = j delta t
    # --------------------------------------------------------
    trajectory = (
        ramsey_trajectory_for_target(
            target_label=target_label,
            times=times,
            cycle_dt_max=
                long_cycle_dt_max,
            shots=shots,
            seed=seed + 100000,
        )
    )

    theta_mods = trajectory[
        "theta_mods"
    ]

    # --------------------------------------------------------
    # Step 4: unwrap phase
    # --------------------------------------------------------
    theta_unwrapped = (
        unwrap_half_angles(
            theta_mods
        )
    )

    # --------------------------------------------------------
    # Step 5: linear fit through the origin for B
    # --------------------------------------------------------
    B_hat_fit = (
        np.sum(
            times
            * theta_unwrapped
        )
        / np.sum(
            times**2
        )
    )

    B_hat_endpoint = (
        theta_unwrapped[-1]
        / times[-1]
    )

    # --------------------------------------------------------
    # Step 6: estimate C
    # --------------------------------------------------------
    C_hat = (
        B_plus_C_hat
        - B_hat_fit
    )

    epsilon_C = (
        epsilon_S
        + epsilon_B
    )

    true_term = TERM_BY_LABEL[
        target_label
    ]

    return {
        "target_label":
            target_label,

        "B_plus_C_hat":
            float(B_plus_C_hat),

        "B_hat_fit":
            float(B_hat_fit),

        "B_hat_endpoint":
            float(B_hat_endpoint),

        "C_hat":
            float(C_hat),

        "epsilon_S":
            float(epsilon_S),

        "epsilon_B":
            float(epsilon_B),

        "epsilon_C":
            float(epsilon_C),

        "sample_dt":
            float(sample_dt),

        "times":
            times,

        "theta_mods":
            theta_mods,

        "theta_unwrapped":
            theta_unwrapped,

        "R_exps":
            trajectory["R_exps"],

        "S_exps":
            trajectory["S_exps"],

        "R_counts":
            trajectory["R_counts"],

        "S_counts":
            trajectory["S_counts"],

        "reference_label":
            trajectory["reference_label"],

        "second_observable_sign":
            trajectory[
                "second_observable_sign"
            ],

        "second_observable_label":
            trajectory[
                "second_observable_label"
            ],

        "reshaping_sequence_length":
            trajectory[
                "reshaping_sequence_length"
            ],

        "long_cycle_dt":
            trajectory["cycle_dt"],

        "true_B":
            float(true_term.B),

        "true_C":
            float(true_term.C),

        "true_omega":
            float(true_term.omega),
    }


# ============================================================
# 12. Learn B and C for all Pauli terms found by
#     the structure-learning stage
# ============================================================
def learn_B_and_C_for_all_targets(
    target_labels: Sequence[str] = LEARNED_SUPPORT,
    T_total: float = 30.0,
    t_short: float = 0.03,
    shots: Optional[int] = None,
) -> Dict[str, Dict[str, object]]:

    results: Dict[
        str,
        Dict[str, object],
    ] = {}

    for index, target_label in enumerate(
        target_labels
    ):
        if not verify_reshaping_sequence(
            target_label
        ):
            raise RuntimeError(
                f"Invalid reshaping sequence for {target_label}."
            )

        result = learn_B_and_C_for_target(
            target_label=target_label,
            T_total=T_total,
            t_short=t_short,
            short_cycle_dt_max=0.003,
            long_cycle_dt_max=0.01,
            safety_factor=0.2,

            # Common prior bounds for all six terms.
            # These are not the exact target parameters.
            B_max=0.8,
            C_max=0.1,
            omega_min=1.0,
            omega_max=1.5,

            shots=shots,
            seed=1000 + 10000 * index,
        )

        results[target_label] = result

    return results


# ============================================================
# 13. Run example
# ============================================================
if __name__ == "__main__":

    results = learn_B_and_C_for_all_targets(
        target_labels=LEARNED_SUPPORT,
        T_total=40.0,
        t_short=0.025,

        # Same convention as the single-qubit example:
        # None means ideal infinite-shot Ramsey measurements.
        shots=None,
    )

    print("=" * 112)
    print(
        "Three-qubit B and C learning after Hamiltonian reshaping"
    )
    print("=" * 112)

    print(
        f"{'P_s':<6}"
        f"{'B true':>12}"
        f"{'B hat':>14}"
        f"{'C true':>12}"
        f"{'C hat':>14}"
        f"{'S hat=B+C':>16}"
        f"{'epsilon_B':>14}"
        f"{'epsilon_C':>14}"
    )

    print("-" * 112)

    for label in LEARNED_SUPPORT:
        r = results[label]

        print(
            f"{label:<6}"
            f"{r['true_B']:>18.15f}"
            f"{r['B_hat_fit']:>18.15f}"
            f"{r['true_C']:>18.15f}"
            f"{r['C_hat']:>18.15f}"
            f"{r['B_plus_C_hat']:>20.15f}"
            f"{r['epsilon_B']:>14.6e}"
            f"{r['epsilon_C']:>14.6e}"
        )

    print("\nRamsey/reshaping details:")

    for label in LEARNED_SUPPORT:
        r = results[label]

        second_sign = (
            "+"
            if r["second_observable_sign"] > 0
            else "-"
        )

        print(
            f"  {label}: "
            f"R={r['reference_label']}, "
            f"S={second_sign}{r['second_observable_label']}, "
            f"L={r['reshaping_sequence_length']}, "
            f"sample_dt={r['sample_dt']:.6f}, "
            f"cycle_dt={r['long_cycle_dt']:.6f}"
        )
