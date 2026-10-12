import numpy as np
import scipy.linalg as la

from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, X as MQ_X, Measure, UnivMathGate
from mindquantum.simulator import Simulator


# ============================================================
# 1. 待学习模型
# H(t) = 0.8X + [0.3 + 0.1cos(9.8t)]Z
# ============================================================
TRUE_A = 0.8
TRUE_B = 0.3
TRUE_C = 0.1
TRUE_OMEGA = 9.8

I2 = np.eye(2, dtype=complex)

X = np.array([
    [0, 1],
    [1, 0]
], dtype=complex)

Z = np.array([
    [1, 0],
    [0, -1]
], dtype=complex)


# ============================================================
# 2. 含时 Trotter 演化
# ============================================================
def trotter_time_evolution(
    t0=0.0,
    tau=0.3,
    trotter_steps=600,
    h_hat=None,
):
    if h_hat is None:
        h_hat = np.zeros((2, 2), dtype=complex)

    dt = tau / trotter_steps
    U_total = I2.copy()

    U_X = la.expm(-1j * TRUE_A * X * dt)
    U_hat = la.expm(+1j * h_hat * dt)

    for k in range(trotter_steps):
        t_mid = t0 + (k + 0.5) * dt
        coeff_z = TRUE_B + TRUE_C * np.cos(TRUE_OMEGA * t_mid)

        U_Z = la.expm(-1j * coeff_z * Z * dt)
        U_step = U_hat @ U_Z @ U_X
        U_total = U_step @ U_total

    return U_total


# ============================================================
# 3. 在 MindQuantum 中执行 Bell 基测量
# ============================================================
def execute_bell_measurement(
    system_unitary,
    shots=5000,
    seed=1234,
):
    if system_unitary.shape != (2, 2):
        raise ValueError("system_unitary 必须是 2*2 矩阵。")

    circuit = Circuit()

    # 制备 |Phi+> = (|00> + |11>)/sqrt(2)
    circuit += H.on(0)
    circuit += MQ_X.on(1, 0)

    # 系统比特 q0 经历完整含时演化，辅助比特 q1 不演化
    U_gate = UnivMathGate("U_system", system_unitary)
    circuit += U_gate.on(0)

    # 逆 Bell 变换：将 Bell 基转换到计算基
    circuit += MQ_X.on(1, 0)
    circuit += H.on(0)

    # 直接测量两个量子比特
    circuit += Measure("m0").on(0)
    circuit += Measure("m1").on(1)

    simulator = Simulator("mqvector", 2, seed=seed)
    measure_result = simulator.sampling(
        circuit,
        shots=shots,
        seed=seed,
    )

    raw_counts = measure_result.data

    # MindQuantum 输出比特串按 m1 m0 排列：
    # 00 -> Phi+ -> I
    # 01 -> Phi- -> Z
    # 10 -> Psi+ -> X
    # 11 -> Psi- -> Y
    bitstring_to_pauli = {
        "00": "I",
        "01": "Z",
        "10": "X",
        "11": "Y",
    }

    pauli_counts = {"I": 0, "X": 0, "Y": 0, "Z": 0}

    for bitstring, count in raw_counts.items():
        pauli_label = bitstring_to_pauli[bitstring]
        pauli_counts[pauli_label] += int(count)

    return pauli_counts, circuit, measure_result


# ============================================================
# 4. 根据实际 Bell 测量计数识别支撑集
# ============================================================
def identify_support_set(
    counts,
    shots,
    minimum_counts=5,
    minimum_frequency=1e-3,
):
    support_set = []

    for label in ("X", "Y", "Z"):
        count = counts[label]
        frequency = count / shots

        if count >= minimum_counts and frequency >= minimum_frequency:
            support_set.append(label)

    return support_set


def main():
    shots = 5000

    U_system = trotter_time_evolution(
        t0=0.0,
        tau=0.3,
        trotter_steps=600,
        h_hat=None,
    )

    counts, circuit, measure_result = execute_bell_measurement(
        system_unitary=U_system,
        shots=shots,
        seed=1234,
    )

    support_set = identify_support_set(
        counts=counts,
        shots=shots,
        minimum_counts=5,
        minimum_frequency=1e-3,
    )

    print("=" * 62)
    print("MindQuantum Bell 基测量")
    print("=" * 62)

    print("\n量子线路：")
    print(circuit)

    print("\n计算基原始测量结果：")
    print(measure_result)

    print("\n转换后的 Bell/Pauli 测量计数：")
    for label in ("I", "X", "Y", "Z"):
        frequency = counts[label] / shots
        print(
            f"  {label}: {counts[label]:5d} 次"
            f"    测量频率 = {frequency:.6f}"
        )

    print(f"\n识别得到的支撑集 S = {support_set}")
    print("理论支撑集           = ['X', 'Z']")
    print("结构学习是否成功     =", set(support_set) == {"X", "Z"})


if __name__ == "__main__":
    main()
