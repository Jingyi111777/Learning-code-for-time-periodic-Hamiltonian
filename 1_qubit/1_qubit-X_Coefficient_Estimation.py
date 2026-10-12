from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# 1. Basic single-qubit matrices
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

KET0 = np.array([1.0, 0.0], dtype=complex)


# ============================================================
# 2. Single-qubit periodically time-dependent Hamiltonian

# ============================================================
@dataclass(frozen=True)
class ModelParameters:
    A: float = 0.8
    B: float = 0.3
    C: float = 0.1
    omega: float = 9.8


TRUE_PARAMS = ModelParameters()


def z_coefficient(
    t: float,
    params: ModelParameters = TRUE_PARAMS,
) -> float:
    return (
        params.B
        + params.C * np.cos(params.omega * t)
    )


def H_single_qubit(
    t: float,
    params: ModelParameters = TRUE_PARAMS,
) -> np.ndarray:
    return (
        params.A * X
        + z_coefficient(t, params) * Z
    )


# ============================================================
# 3. Fast 2x2 evolution for h_x X + h_z Z
# ============================================================
def pauli_xz_step(
    h_x: float,
    h_z: float,
    dt: float,
) -> np.ndarray:
    radius = float(np.sqrt(h_x**2 + h_z**2))

    if radius < 1e-15:
        return I2.copy()

    H_axis = h_x * X + h_z * Z

    return (
        np.cos(radius * dt) * I2
        - 1j
        * np.sin(radius * dt)
        / radius
        * H_axis
    )


# ============================================================
# 4. Hamiltonian reshaping for the target X term

# ============================================================
def apply_one_x_reshaping_cycle(
    state: np.ndarray,
    t_left: float,
    cycle_dt: float,
    params: ModelParameters = TRUE_PARAMS,
) -> np.ndarray:
    out = np.asarray(state, dtype=complex).reshape(2)

    # One deterministic reshaping cycle uses Q = I and Q = X.
    sub_dt = cycle_dt / 2.0

    # First half: Q = I.
    t_mid_1 = t_left + 0.5 * sub_dt
    hz_1 = z_coefficient(t_mid_1, params)

    U_1 = pauli_xz_step(
        h_x=params.A,
        h_z=hz_1,
        dt=sub_dt,
    )
    out = U_1 @ out

    # Second half: Q = X.
    # Conjugation by X preserves X and flips the sign of Z.
    t_mid_2 = t_left + 1.5 * sub_dt
    hz_2 = -z_coefficient(t_mid_2, params)

    U_2 = pauli_xz_step(
        h_x=params.A,
        h_z=hz_2,
        dt=sub_dt,
    )
    out = U_2 @ out

    return out


def apply_x_reshaped_evolution(
    initial_state: np.ndarray,
    T_total: float,
    cycle_dt_max: float = 0.002,
    params: ModelParameters = TRUE_PARAMS,
) -> np.ndarray:
    if T_total < 0:
        raise ValueError("T_total must be nonnegative.")

    if cycle_dt_max <= 0:
        raise ValueError("cycle_dt_max must be positive.")

    out = np.asarray(
        initial_state,
        dtype=complex,
    ).reshape(2)

    if T_total == 0:
        return out.copy()

    n_cycles = max(
        1,
        int(np.ceil(T_total / cycle_dt_max)),
    )

    cycle_dt = T_total / n_cycles
    current_time = 0.0

    for _ in range(n_cycles):
        out = apply_one_x_reshaping_cycle(
            state=out,
            t_left=current_time,
            cycle_dt=cycle_dt,
            params=params,
        )
        current_time += cycle_dt

    norm = np.linalg.norm(out)

    if norm <= 0:
        raise RuntimeError("The evolved state has zero norm.")

    return out / norm


# ============================================================
# 5. Ramsey signal for the isolated X coefficient

# ============================================================
def pauli_expectation(
    state: np.ndarray,
    observable: np.ndarray,
) -> float:
    psi = np.asarray(
        state,
        dtype=complex,
    ).reshape(2)

    value = np.real(
        np.vdot(
            psi,
            observable @ psi,
        )
    )

    return float(
        np.clip(value, -1.0, 1.0)
    )


def sample_pauli_expectation(
    exact_expectation: float,
    shots: Optional[int],
    rng: np.random.Generator,
) -> float:
    if shots is None:
        return float(exact_expectation)

    if shots <= 0:
        raise ValueError("shots must be positive or None.")

    p_plus = 0.5 * (
        1.0 + exact_expectation
    )

    n_plus = int(
        rng.binomial(
            shots,
            p_plus,
        )
    )

    n_minus = shots - n_plus

    return float(
        (n_plus - n_minus) / shots
    )


def ramsey_signal_for_x(
    T_total: float,
    cycle_dt_max: float = 0.002,
    shots: Optional[int] = None,
    seed: int = 0,
    params: ModelParameters = TRUE_PARAMS,
) -> Dict[str, object]:
    final_state = apply_x_reshaped_evolution(
        initial_state=KET0,
        T_total=T_total,
        cycle_dt_max=cycle_dt_max,
        params=params,
    )

    z_exact = pauli_expectation(
        final_state,
        Z,
    )

    y_exact = pauli_expectation(
        final_state,
        Y,
    )

    rng = np.random.default_rng(seed)

    z_measured = sample_pauli_expectation(
        exact_expectation=z_exact,
        shots=shots,
        rng=rng,
    )

    y_measured = sample_pauli_expectation(
        exact_expectation=y_exact,
        shots=shots,
        rng=rng,
    )

    # For ideal H_eff = A X:
    # Z - iY = exp(i 2 A t).
    S_hat = (
        z_measured
        - 1j * y_measured
    )

    magnitude = abs(S_hat)

    if magnitude > 1e-15:
        S_hat = S_hat / magnitude

    return {
        "T_total": float(T_total),
        "Z_exact": float(z_exact),
        "Y_exact": float(y_exact),
        "Z_measured": float(z_measured),
        "Y_measured": float(y_measured),
        "S_hat": complex(S_hat),
    }


# ============================================================
# 6. Robust interval refinement for the X coefficient
# ============================================================
def learn_x_coefficient(
    epsilon: float = 0.01,
    a_init: float = -1.0,
    b_init: float = 1.0,
    cycle_dt_max: float = 0.002,
    shots: Optional[int] = None,
    seed: int = 1234,
    params: ModelParameters = TRUE_PARAMS,
) -> Dict[str, object]:
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")

    if not a_init < b_init:
        raise ValueError("Require a_init < b_init.")

    a = float(a_init)
    b = float(b_init)

    history: List[Dict[str, float]] = []
    iteration = 0

    while (b - a) > 2.0 * epsilon:
        iteration += 1

        width = b - a
        center = 0.5 * (a + b)

        # Robust-frequency-estimation interrogation time.
        tau = np.pi / (
            2.0 * width
        )

        ramsey = ramsey_signal_for_x(
            T_total=tau,
            cycle_dt_max=cycle_dt_max,
            shots=shots,
            seed=seed + iteration,
            params=params,
        )

        S_hat = complex(
            ramsey["S_hat"]
        )

        # Remove the phase predicted at the center of [a,b].
        center_phase = (
            2.0 * center * tau
        )

        relative_phase = np.angle(
            S_hat
            * np.exp(
                -1j * center_phase
            )
        )

        
        phase_estimate = (
            center
            + relative_phase
            / (2.0 * tau)
        )

        old_a = a
        old_b = b

        
        if phase_estimate <= center:
            b = old_a + (
                2.0 / 3.0
            ) * width
            decision = "left"
        else:
            a = old_a + (
                1.0 / 3.0
            ) * width
            decision = "right"

        history.append({
            "iteration": float(iteration),
            "a_before": float(old_a),
            "b_before": float(old_b),
            "tau": float(tau),
            "Z_measured": float(
                ramsey["Z_measured"]
            ),
            "Y_measured": float(
                ramsey["Y_measured"]
            ),
            "phase_estimate": float(
                phase_estimate
            ),
            "a_after": float(a),
            "b_after": float(b),
            "decision": decision,
        })

    A_hat = 0.5 * (a + b)

    return {
        "A_hat": float(A_hat),
        "true_A": float(params.A),
        "absolute_error": float(
            abs(A_hat - params.A)
        ),
        "final_interval": (
            float(a),
            float(b),
        ),
        "epsilon": float(epsilon),
        "iterations": int(iteration),
        "shots": shots,
        "cycle_dt_max": float(
            cycle_dt_max
        ),
        "history": history,
    }


# ============================================================
# 7. Run example
# ============================================================
if __name__ == "__main__":
    result = learn_x_coefficient(
        epsilon=0.01,
        a_init=-1.0,
        b_init=1.0,
        cycle_dt_max=0.002,

        # None gives ideal expectation values.
        # Replace by, e.g., 2000 for finite-shot simulation.
        shots=None,

        seed=1234,
        params=TRUE_PARAMS,
    )

    print("=" * 78)
    print(
        "Learning the static X coefficient in "
        "H(t)=0.8 X + [0.3+0.1 cos(9.8 t)] Z"
    )
    print("=" * 78)

    print(
        f"True X coefficient A : "
        f"{result['true_A']:.10f}"
    )

    print(
        f"Estimated A_hat      : "
        f"{result['A_hat']:.10f}"
    )

    print(
        f"Absolute error       : "
        f"{result['absolute_error']:.10e}"
    )

    print(
        f"Final interval       : "
        f"[{result['final_interval'][0]:.10f}, "
        f"{result['final_interval'][1]:.10f}]"
    )

    print(
        f"Iterations           : "
        f"{result['iterations']}"
    )

    print("\nInterval-refinement history:")

    print(
        f"{'iter':>5}"
        f"{'[a,b] before':>28}"
        f"{'tau':>12}"
        f"{'A phase':>14}"
        f"{'branch':>10}"
        f"{'[a,b] after':>28}"
    )

    print("-" * 97)

    for row in result["history"]:
        before = (
            f"[{row['a_before']:.6f},"
            f"{row['b_before']:.6f}]"
        )

        after = (
            f"[{row['a_after']:.6f},"
            f"{row['b_after']:.6f}]"
        )

        print(
            f"{int(row['iteration']):>5d}"
            f"{before:>28}"
            f"{row['tau']:>12.6f}"
            f"{row['phase_estimate']:>14.8f}"
            f"{row['decision']:>10}"
            f"{after:>28}"
        )
