import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import scipy.linalg as la

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, X as MQ_X, Measure, UnivMathGate
from mindquantum.simulator import Simulator


# ============================================================
# 1. Basic Pauli matrices
# ============================================================
I2 = np.eye(2, dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Y = np.array([
    [0, -1j],
    [1j, 0]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
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


# ============================================================
# 3. Periodically time-dependent three-qubit Hamiltonian
#
# H(t) =
# [0.72 + 0.08 cos(1.0 t)] XII
# + [0.59 + 0.06 cos(1.1 t)] ZZI
# + [0.36 + 0.04 cos(1.2 t)] IXX
# + [0.29 + 0.03 cos(1.3 t)] YIZ
# + [0.18 + 0.02 cos(1.4 t)] XYZ
# + [0.145+0.015 cos(1.5 t)] ZXY
#
# At t=0:
# Layer 1: 0.80, 0.65
# Layer 2: 0.40, 0.32
# Layer 3: 0.20, 0.16
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

TRUE_SUPPORT = [
    term.label
    for term in TERMS
]

OMEGA_MAX = max(
    term.omega
    for term in TERMS
)


def H_true(t: float) -> np.ndarray:
    """Full unknown Hamiltonian H(t)."""
    H_mat = np.zeros((DIM_SYSTEM, DIM_SYSTEM), dtype=complex)

    for term in TERMS:
        H_mat += (
            term.coefficient(t)
            * ALL_PAULI_MATRICES[term.label]
        )

    return H_mat


def H_learned_at_reference_time(
    t0: float,
    learned_labels: Iterable[str],
) -> np.ndarray:
    """
    Construct the already-learned Hamiltonian at the
    reference time t0.

    In this structure-learning demonstration, previously
    learned terms are cancelled using their exact model
    coefficients at t0. In a complete learning pipeline,
    these coefficients would be replaced by their estimates.
    """
    H_hat = np.zeros((DIM_SYSTEM, DIM_SYSTEM), dtype=complex)

    for label in learned_labels:
        term = TERM_BY_LABEL[label]
        H_hat += (
            term.coefficient(t0)
            * ALL_PAULI_MATRICES[label]
        )

    return H_hat


# ============================================================
# 4. Residual time evolution
# exp(+i H_hat dt) exp(-i H(t_mid) dt).
# ============================================================
def trotter_residual_time_evolution(
    learned_labels: Sequence[str],
    t0: float,
    tau: float,
    trotter_steps: int,
) -> np.ndarray:

    if tau <= 0:
        raise ValueError("tau must be positive.")

    if trotter_steps <= 0:
        raise ValueError("trotter_steps must be positive.")

    dt = tau / trotter_steps

    U_total = np.eye(DIM_SYSTEM, dtype=complex)

    H_hat = H_learned_at_reference_time(
        t0=t0,
        learned_labels=learned_labels,
    )

    U_hat_inverse = la.expm(
        +1j * H_hat * dt
    )

    for k in range(trotter_steps):
        t_mid = t0 + (k + 0.5) * dt

        U_true_step = la.expm(
            -1j * H_true(t_mid) * dt
        )

        U_step = (
            U_hat_inverse
            @ U_true_step
        )

        U_total = (
            U_step
            @ U_total
        )

    return U_total


# ============================================================
# 5. Explicit six-qubit Bell circuit
# ============================================================
SYSTEM_QUBITS = [0, 1, 2]
ANCILLA_QUBITS = [3, 4, 5]
N_TOTAL_QUBITS = 6


def execute_six_qubit_bell_measurement(
    system_unitary: np.ndarray,
    shots: int = 200_000,
    seed: int = 1234,
):
    
    if system_unitary.shape != (8, 8):
        raise ValueError(
            "system_unitary must be an 8 x 8 matrix."
        )

    circuit = Circuit()

    # --------------------------------------------------------
    # Step 1: prepare three Bell pairs
    # --------------------------------------------------------
    for q_sys, q_anc in zip(
        SYSTEM_QUBITS,
        ANCILLA_QUBITS,
    ):
        circuit += H.on(q_sys)
        circuit += MQ_X.on(q_anc, q_sys)

    # --------------------------------------------------------
    # Step 2: unknown/residual three-qubit evolution
    # --------------------------------------------------------
    U_gate = UnivMathGate(
        "U_residual",
        system_unitary,
    )

    circuit += U_gate.on(SYSTEM_QUBITS)

    # --------------------------------------------------------
    # Step 3: inverse Bell transformation on each pair
    # --------------------------------------------------------
    for q_sys, q_anc in zip(
        SYSTEM_QUBITS,
        ANCILLA_QUBITS,
    ):
        circuit += MQ_X.on(q_anc, q_sys)
        circuit += H.on(q_sys)

    # --------------------------------------------------------
    # Step 4: measure all six qubits
    # --------------------------------------------------------
    for q in range(N_TOTAL_QUBITS):
        circuit += Measure(f"m{q}").on(q)

    simulator = Simulator(
        "mqvector",
        N_TOTAL_QUBITS,
        seed=seed,
    )

    measure_result = simulator.sampling(
        circuit,
        shots=shots,
        seed=seed,
    )

    raw_counts = measure_result.data

    return (
        raw_counts,
        circuit,
        measure_result,
    )


# ============================================================
# 6. Convert six measured bits into a three-qubit Pauli label
# ============================================================
PAIR_BITS_TO_PAULI = {
    "00": "I",
    "01": "Z",
    "10": "X",
    "11": "Y",
}


def get_measurement_bit(
    bitstring: str,
    measurement_index: int,
) -> str:
    
    if len(bitstring) != N_TOTAL_QUBITS:
        raise ValueError(
            f"Expected a {N_TOTAL_QUBITS}-bit string, "
            f"got {bitstring}"
        )

    return bitstring[
        N_TOTAL_QUBITS - 1 - measurement_index
    ]


def six_bitstring_to_pauli_label(
    bitstring: str,
) -> str:
    
    pauli_chars: List[str] = []

    for i in range(3):
        system_bit = get_measurement_bit(
            bitstring,
            i,
        )

        ancilla_bit = get_measurement_bit(
            bitstring,
            i + 3,
        )

        pair_bits = (
            ancilla_bit
            + system_bit
        )

        pauli_chars.append(
            PAIR_BITS_TO_PAULI[pair_bits]
        )

    return "".join(pauli_chars)


def convert_raw_counts_to_pauli_counts(
    raw_counts: Dict[str, int],
) -> Dict[str, int]:
    
    pauli_counts = {
        label: 0
        for label in ALL_PAULI_LABELS
    }

    for bitstring, count in raw_counts.items():
        label = six_bitstring_to_pauli_label(
            bitstring
        )

        pauli_counts[label] += int(count)

    return pauli_counts


# ============================================================
# 7. Identify the current hierarchy layer
# ============================================================
def identify_current_layer(
    pauli_counts: Dict[str, int],
    shots: int,
    tau: float,
    mu_lower: float,
    already_learned: Sequence[str],
    minimum_counts: int = 5,
    threshold_scale: float = 1.0,
) -> Tuple[List[str], float]:
    
    probability_threshold = (
        threshold_scale
        * (mu_lower * tau) ** 2
    )

    learned_set = set(already_learned)

    identified: List[str] = []

    for label in ALL_PAULI_LABELS:
        if label == "III":
            continue

        if label in learned_set:
            continue

        count = pauli_counts[label]
        frequency = count / shots

        if (
            count >= minimum_counts
            and frequency >= probability_threshold
        ):
            identified.append(label)

    identified.sort(
        key=lambda label: pauli_counts[label],
        reverse=True,
    )

    return (
        identified,
        probability_threshold,
    )


# ============================================================
# 8. Define the parameters for the three-layer hierarchical structure-learning procedure.
#Print the Pauli terms with the highest occurrence counts in the Bell-sampling results.
# ============================================================
LAYER_SPECS = (
    {
        "name": "Layer 1",
        "mu_lower": 0.5,
        "tau": 0.04,
        "trotter_steps": 400,
        "expected": ["XII", "ZZI"],
    },
    {
        "name": "Layer 2",
        "mu_lower": 0.25,
        "tau": 0.06,
        "trotter_steps": 600,
        "expected": ["IXX", "YIZ"],
    },
    {
        "name": "Layer 3",
        "mu_lower": 0.125,
        "tau": 0.08,
        "trotter_steps": 800,
        "expected": ["XYZ", "ZXY"],
    },
)


def print_top_pauli_counts(
    pauli_counts: Dict[str, int],
    shots: int,
    top_k: int = 10,
):
    data = []

    for label, count in pauli_counts.items():
        if label == "III":
            continue

        data.append(
            (
                label,
                count,
                count / shots,
            )
        )

    data.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    for label, count, frequency in data[:top_k]:
        print(
            f"    {label}: "
            f"{count:7d} counts"
            f"    frequency = {frequency:.8f}"
        )


# ============================================================
# 9. Complete three-layer structure-learning demonstration
# ============================================================
def run_hierarchical_structure_learning(
    shots: int = 200_000,
    t0: float = 0.0,
    seed: int = 1234,
):
    learned_labels: List[str] = []
    layer_results = []

    print("=" * 78)
    print(
        "Three-qubit hierarchical structure learning "
        "with an explicit six-qubit Bell circuit"
    )
    print("=" * 78)

    print("\nTrue Hamiltonian at t = 0:")
    for term in TERMS:
        print(
            f"  {term.label}: "
            f"B={term.B:.3f}, "
            f"C={term.C:.3f}, "
            f"omega={term.omega:.3f}, "
            f"mu(0)={term.coefficient(0.0):.3f}"
        )

    for layer_index, spec in enumerate(
        LAYER_SPECS,
        start=1,
    ):
        tau = float(spec["tau"])
        trotter_steps = int(
            spec["trotter_steps"]
        )
        mu_lower = float(
            spec["mu_lower"]
        )

        # ----------------------------------------------------
        # Residual system evolution for this hierarchy layer
        # ----------------------------------------------------
        U_residual = trotter_residual_time_evolution(
            learned_labels=learned_labels,
            t0=t0,
            tau=tau,
            trotter_steps=trotter_steps,
        )

        # ----------------------------------------------------
        # EXPLICIT 6-QUBIT BELL CIRCUIT
        # ----------------------------------------------------
        raw_counts, circuit, measure_result = (
            execute_six_qubit_bell_measurement(
                system_unitary=U_residual,
                shots=shots,
                seed=seed + layer_index,
            )
        )

        pauli_counts = (
            convert_raw_counts_to_pauli_counts(
                raw_counts
            )
        )

        identified, probability_threshold = (
            identify_current_layer(
                pauli_counts=pauli_counts,
                shots=shots,
                tau=tau,
                mu_lower=mu_lower,
                already_learned=learned_labels,
                minimum_counts=5,
                threshold_scale=1.0,
            )
        )

        expected = list(
            spec["expected"]
        )

        success = (
            set(identified)
            == set(expected)
        )

        print("\n" + "-" * 78)
        print(spec["name"])
        print("-" * 78)

        print(
            f"  learning window tau        = {tau}"
        )
        print(
            f"  Trotter steps              = {trotter_steps}"
        )
        print(
            f"  max(omega_s) * tau         = "
            f"{OMEGA_MAX * tau:.6f}"
        )
        print(
            f"  coefficient lower bound    = {mu_lower}"
        )
        print(
            f"  Bell-frequency threshold   = "
            f"{probability_threshold:.8e}"
        )
        print(
            f"  previously learned terms   = "
            f"{learned_labels}"
        )

        
        if layer_index == 1:
            print("\nSix-qubit Bell circuit:")
            print(circuit)

        print(
            "\nTop nonidentity Bell/Pauli outcomes:"
        )
        print_top_pauli_counts(
            pauli_counts=pauli_counts,
            shots=shots,
            top_k=10,
        )

        print(
            f"\n  identified support         = "
            f"{identified}"
        )
        print(
            f"  expected support           = "
            f"{expected}"
        )
        print(
            f"  layer learning successful  = "
            f"{success}"
        )

        layer_results.append(
            {
                "layer": layer_index,
                "name": spec["name"],
                "tau": tau,
                "trotter_steps": trotter_steps,
                "mu_lower": mu_lower,
                "probability_threshold":
                    probability_threshold,
                "raw_counts": raw_counts,
                "pauli_counts": pauli_counts,
                "identified_support": identified,
                "expected_support": expected,
                "success": success,
                "circuit": circuit,
                "measure_result": measure_result,
                "learned_before":
                    list(learned_labels),
            }
        )

        
        learned_labels.extend(
            identified
        )

    total_success = (
        set(learned_labels)
        == set(TRUE_SUPPORT)
    )

    print("\n" + "=" * 78)
    print("Final result")
    print("=" * 78)

    print(
        f"  learned Pauli support = "
        f"{learned_labels}"
    )
    print(
        f"  true Pauli support    = "
        f"{TRUE_SUPPORT}"
    )
    print(
        f"  total learning success = "
        f"{total_success}"
    )

    return {
        "learned_support":
            learned_labels,
        "true_support":
            TRUE_SUPPORT,
        "success":
            total_success,
        "layer_results":
            layer_results,
        "shots_per_layer":
            shots,
        "t0":
            t0,
    }


# ============================================================
# 10. Run example
# ============================================================
if __name__ == "__main__":
    result = run_hierarchical_structure_learning(
        shots=200_000,
        t0=0.0,
        seed=1234,
    )
