import numpy as np
import scipy.linalg as la
from dataclasses import dataclass
from typing import Dict, Optional


# ============================================================
# Basic gates
# ============================================================

I2 = np.eye(2, dtype=complex)

X = np.array([[0, 1],
              [1, 0]], dtype=complex)

Y = np.array([[0, -1j],
              [1j, 0]], dtype=complex)

Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

HAD = (1 / np.sqrt(2)) * np.array([[1, 1],
                                   [1, -1]], dtype=complex)


S = np.array([[1, 0],
              [0, 1j]], dtype=complex)

SDG = np.array([[1, 0],
                [0, -1j]], dtype=complex)


# ============================================================
# H(t) = x_coeff X + [B + C cos(omega t)] Z
# ============================================================

@dataclass
class HamiltonianParams:
    x_coeff: float = 0.8
    B: float = 0.3
    C: float = 0.1
    omega: float = 9.8


DEFAULT_PARAMS = HamiltonianParams(
    x_coeff=0.8,
    B=0.3,
    C=0.1,
    omega=9.8,
)


def H_single_qubit(t: float, params: HamiltonianParams) -> np.ndarray:
    """
    H(t) = x_coeff X + [B + C cos(omega t)] Z.
    """
    return (
        params.x_coeff * X
        +
        (params.B + params.C * np.cos(params.omega * t)) * Z
    )


def apply_probe_gate(state: np.ndarray, gate: np.ndarray) -> np.ndarray:
    
    return state @ gate.T


# ============================================================
# 1. Reshaped evolution without ESU
# ============================================================

def apply_reshaped_no_esu_state(
    probe_state: np.ndarray,
    T_total: float,
    dt_max: float = 0.03,
    scale: float = 1.0,
    params: HamiltonianParams = DEFAULT_PARAMS,
) -> np.ndarray:
    
    state = np.asarray(probe_state, dtype=complex).reshape(1, 2)
    """总演化时间 [0,Ttotal] 分成很多小段"""
    bounds = sorted(
        set(
            round(float(t), 12)
            for t in [0.0, T_total] + list(np.arange(0.0, T_total + 1e-12, dt_max))
        )
    )

    out = state.copy()

    for left, right in zip(bounds[:-1], bounds[1:]):
        segment_dt = right - left
        if segment_dt <= 1e-14:
            continue

        micro_dt = segment_dt / 2.0

        # First half evolution
        t_mid_1 = left + 0.5 * micro_dt
        U1 = la.expm(-1j * H_single_qubit(t_mid_1, params) * scale * micro_dt)
        out = apply_probe_gate(out, U1)
        out = apply_probe_gate(out, Z)

        # Second half evolution
        t_mid_2 = left + micro_dt + 0.5 * micro_dt
        U2 = la.expm(-1j * H_single_qubit(t_mid_2, params) * scale * micro_dt)
        out = apply_probe_gate(out, U2)
        out = apply_probe_gate(out, Z)

    return out.reshape(2)


# ============================================================
# 2. Ramsey measurement through Z-basis readout
# ============================================================

def measure_z_after_rotation(
    state: np.ndarray,
    rotation: np.ndarray,
    shots: Optional[int] = 1000,
    seed: int = 0,
) -> Dict[str, object]:
    
    state = np.asarray(state, dtype=complex).reshape(2)

    # Rotate the target Pauli basis to the computational Z basis.
    rotated = rotation @ state

    probs = np.abs(rotated) ** 2
    probs = probs / np.sum(probs)

    if shots is None:
        # Ideal infinite-shot limit, still using measurement probabilities.
        exp_val = float(probs[0] - probs[1])
        counts = None
    else:
        rng = np.random.default_rng(seed)
        samples = rng.choice([0, 1], size=shots, p=probs)
        counts = np.bincount(samples, minlength=2)

        # outcome 0 gives +1, outcome 1 gives -1
        exp_val = float((counts[0] - counts[1]) / shots)

    return {
        "exp_val": exp_val,
        "probs": probs,
        "counts": counts,
    }


def ramsey_expectations_for_B(
    t: float,
    dt_max: float = 0.03,
    params: HamiltonianParams = DEFAULT_PARAMS,
    shots: Optional[int] = 1000,
    seed: int = 0,
) -> Dict[str, object]:
   

    # 1. Prepare |+> from |0>.
    zero = np.array([1.0, 0.0], dtype=complex)
    plus = HAD @ zero

    # 2. Apply reshaped evolution without ESU.
    final = apply_reshaped_no_esu_state(
        plus,
        T_total=t,
        dt_max=dt_max,
        scale=1.0,
        params=params,
    )

    # 3. X-Pauli measurement through Z-basis readout.
    #    HAD^dagger Z HAD = X.
    x_meas = measure_z_after_rotation(
        final,
        rotation=HAD,
        shots=shots,
        seed=seed,
    )
    ex = float(x_meas["exp_val"])

    # 4. Y-Pauli measurement through Z-basis readout.
    #    (HAD @ SDG)^dagger Z (HAD @ SDG) = Y.
    y_rotation = HAD @ SDG
    y_meas = measure_z_after_rotation(
        final,
        rotation=y_rotation,
        shots=shots,
        seed=seed + 100000,
    )
    ey = float(y_meas["exp_val"])

    # 5. Recover Ramsey phase.
    theta_mod = 0.5 * np.arctan2(ey, ex)

    return {
        "X_exp": ex,
        "Y_exp": ey,
        "theta_mod": float(theta_mod),
        "X_counts": x_meas["counts"],
        "Y_counts": y_meas["counts"],
        "X_probs": x_meas["probs"],
        "Y_probs": y_meas["probs"],
    }


# ============================================================
# 3. Unwrap theta measured modulo pi
# ============================================================

def unwrap_half_angles(theta_mods):
    
    theta_mods = np.asarray(theta_mods, dtype=float)
    return 0.5 * np.unwrap(2.0 * theta_mods)


# ============================================================
# 4. Short-time rough estimate of B + C
# ============================================================

def estimate_B_plus_C_short_time(
    t_short: float = 0.03,
    dt_max: float = 0.003,
    C_max: float = 0.12,
    omega_max: float = 11.0,
    params: HamiltonianParams = DEFAULT_PARAMS,
    shots: Optional[int] = None,
) -> Dict[str, float]:

    result = ramsey_expectations_for_B(
        t=t_short,
        dt_max=dt_max,
        params=params,
        shots=shots,
        seed=123,
    )

    theta_short = result["theta_mod"]
    B_plus_C_hat = theta_short / t_short

    # Theoretical short-time error bound from the paper:
    # epsilon_S = C_max * omega_max^2 * t_short^2 / 6
    epsilon_S = C_max * omega_max**2 * t_short**2 / 6.0

    return {
        "B_plus_C_hat": float(B_plus_C_hat),
        "theta_short": float(theta_short),
        "epsilon_S": float(epsilon_S),
    }


# ============================================================
# 5. Long-time fitting for B
# ============================================================

def learn_B_by_long_time_average_no_esu(
    T_total: float = 30.0,
    dt_max: float = 0.03,
    safety_factor: float = 0.2,
    t_short: float = 0.03,
    B_max: float = 0.3,
    C_max: float = 0.12,
    omega_min: float = 8.5,
    omega_max: float = 11.0,
    params: HamiltonianParams = DEFAULT_PARAMS,
    shots: Optional[int] = None,
) -> Dict[str, object]:

    # --------------------------------------------------------
    # Step 1: short-time rough estimate of B + C
    # --------------------------------------------------------
    short_result = estimate_B_plus_C_short_time(
        t_short=t_short,
        dt_max=0.003,
        C_max=C_max,
        omega_max=omega_max,
        params=params,
        shots=shots,
    )

    B_plus_C_hat = short_result["B_plus_C_hat"]
    epsilon_S = short_result["epsilon_S"]

    # --------------------------------------------------------
    # Step 2: choose sampling interval dt
    # Requirement:
    #     |B + C| dt < pi / 2.
    # --------------------------------------------------------
    if B_max + C_max <= 0:
        raise ValueError(
            "B_max + C_max must be positive."
        )

    sample_dt = safety_factor * np.pi / (2.0 * (B_max + C_max))

    times = np.arange(sample_dt, T_total + 1e-12, sample_dt)

    N = len(times)
    epsilon_B = (
        3.0 * C_max
        / (omega_min * sample_dt * (2 * N + 1))
    )

    # Since C_hat = S_hat - B_hat:
    # epsilon_C = epsilon_S + epsilon_B
    epsilon_C = epsilon_S + epsilon_B

    # --------------------------------------------------------
    # Step 3: measure theta(t_j)
    # --------------------------------------------------------
    theta_mods = []
    X_exps = []
    Y_exps = []
    X_counts = []
    Y_counts = []
    X_probs = []
    Y_probs = []

    for j, t in enumerate(times):
        result = ramsey_expectations_for_B(
            t=float(t),
            dt_max=dt_max,
            params=params,
            shots=shots,
            seed=1000 + j,
        )

        theta_mods.append(result["theta_mod"])
        #保存当前时间点的 X、Y 期望值
        X_exps.append(result["X_exp"])
        Y_exps.append(result["Y_exp"])
        #保存有限 shots 情况下的测量计数
        X_counts.append(result["X_counts"])
        Y_counts.append(result["Y_counts"])
        X_probs.append(result["X_probs"])
        Y_probs.append(result["Y_probs"])

    theta_mods = np.asarray(theta_mods)
    X_exps = np.asarray(X_exps)
    Y_exps = np.asarray(Y_exps)
    X_probs = np.asarray(X_probs)
    Y_probs = np.asarray(Y_probs)

    # If shots is None, counts are None. Keep them as object arrays.
    X_counts = np.asarray(X_counts, dtype=object)
    Y_counts = np.asarray(Y_counts, dtype=object)

    # --------------------------------------------------------
    # Step 4: unwrap phase
    # --------------------------------------------------------
    theta_unwrapped = unwrap_half_angles(theta_mods)

    # --------------------------------------------------------
    # Step 5A: long-time linear fit
    # --------------------------------------------------------
    B_hat_fit = np.sum(times * theta_unwrapped) / np.sum(times**2)
    B_hat_endpoint = theta_unwrapped[-1] / times[-1]

    C_hat = B_plus_C_hat - B_hat_fit

    return {
        "B_hat_fit": float(B_hat_fit),
        "B_hat_endpoint": float(B_hat_endpoint),
        "B_plus_C_hat": float(B_plus_C_hat),
        "C_hat": float(C_hat),
        "epsilon_S": float(epsilon_S),
        "epsilon_B": float(epsilon_B),
        "epsilon_C": float(epsilon_C),
        "B_max": float(B_max),
        "C_max": float(C_max),
        "omega_min": float(omega_min),
        "omega_max": float(omega_max),
        "t_short": float(t_short),
        "sample_dt": float(sample_dt),
        "times": times,
        "theta_mods": theta_mods,
        "theta_unwrapped": theta_unwrapped,
        "X_exps": X_exps,
        "Y_exps": Y_exps,
        "X_counts": X_counts,
        "Y_counts": Y_counts,
        "X_probs": X_probs,
        "Y_probs": Y_probs,
    }
# ============================================================
# 6. Run example
# ============================================================

if __name__ == "__main__":
    result = learn_B_by_long_time_average_no_esu(
        T_total=30.0,
        dt_max=0.03,
        safety_factor=0.2,
        t_short=0.03,
        B_max=0.3,
        C_max=0.12,      # prior upper bound; not the true unknown C
        omega_min=8.5,
        omega_max=11.0,
        params=DEFAULT_PARAMS,
        shots=None,
    )

    
    print("Prior B_max                  =", result["B_max"])
    print("Prior C_max                  =", result["C_max"])
    print("Prior omega_min              =", result["omega_min"])
    print("Prior omega_max              =", result["omega_max"])
    print("Short-time estimate of B + C =", result["B_plus_C_hat"])
    print("Chosen sample dt             =", result["sample_dt"])

    print("Estimated B by long-time fit =", result["B_hat_fit"])
    print("Estimated B by theta(T)/T    =", result["B_hat_endpoint"])
    print("Estimated C                   =", result["C_hat"])
    print("Theoretical epsilon_S        =", result["epsilon_S"])
    print("Theoretical epsilon_B        =", result["epsilon_B"])
    print("Theoretical epsilon_C        =", result["epsilon_C"])
