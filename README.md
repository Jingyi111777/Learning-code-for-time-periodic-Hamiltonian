# Learning Periodically Time-Dependent Quantum Hamiltonians

Numerical examples for learning the structure and parameters of
periodically time-dependent Hamiltonians in single-qubit and
three-qubit systems.

The examples cover Pauli-support identification, coefficient estimation,
adaptive frequency estimation, and coefficient refinement.

## 1. Hamiltonian Models

### Single-qubit model

$$
H(t)=AX+\left[B+C\cos(\omega t)\right]Z.
$$

The default simulation parameters are:

| Parameter | Value |
|---|---:|
| A | 0.8 |
| B | 0.3 |
| C | 0.1 |
| ω | 9.8 |

### Three-qubit model

$$
H(t)=\sum_{s\in S}\left[B_s+C_s\cos(\omega_s t)\right]P_s,
$$

where each $P_s$ is a three-qubit Pauli string.

The default simulation parameters are:

| Pauli string | B | C | ω |
|---|---:|---:|---:|
| XII | 0.72 | 0.08 | 1.0 |
| ZZI | 0.59 | 0.06 | 1.1 |
| IXX | 0.36 | 0.04 | 1.2 |
| YIZ | 0.29 | 0.03 | 1.3 |
| XYZ | 0.18 | 0.02 | 1.4 |
| ZXY | 0.145 | 0.015 | 1.5 |

These parameters define the simulated systems. True parameters are also
used to evaluate estimation errors and interval coverage.

## 2. Files

### Single-qubit examples

| File | Purpose |
|---|---|
| `1 qubit -Structure Learning.py` | Identify the Pauli support using Bell-basis measurements. |
| `1 qubit -X_Coefficient_Estimation.py` | Estimate the static X coefficient A using Hamiltonian reshaping and iterative interval refinement. |
| `1 qubit- B and C Estimation(3).py` | Obtain preliminary estimates of B and C from long-time and short-time Ramsey signals. |
| `1 qubit_omega_adaptive.py` | Estimate ω using an ESU sensing response, quantum signal processing (QSP), and BBHT amplitude search. |
| `1 qubit_BC_refinement.py` | Refine the B and C intervals using the frequency estimate and its error bound. |

### Three-qubit examples

| File | Purpose |
|---|---|
| `3 qubits-Structure Learning(1).py` | Perform hierarchical Pauli-support learning using Bell-basis measurements. |
| `3 qubits-B and C estimate(1).py` | Construct target-selective reshaping sequences and estimate B and C for each target term. |
| `3 qubit_omega_adaptive(1).py` | Design and simulate adaptive QSP/BBHT frequency estimation for each target term. |
| `3 qubit_BC_refinement(1).py` | Refine B and C for all target terms using preliminary coefficient and frequency intervals. |

## 3. Dependencies

The scripts use:

- Python
- NumPy
- SciPy
- MindQuantum, for the structure-learning examples

Install the dependencies in a suitable Python environment:

```bash
python -m pip install numpy scipy mindquantum
```

The NumPy/SciPy examples can be run without MindQuantum:

```bash
python -m pip install numpy scipy
```

## 4. Running the Examples

Run the commands from the directory containing the scripts.
Quotation marks are required because the filenames contain spaces.

### Single-qubit workflow

```bash
python "1 qubit -Structure Learning.py"
python "1 qubit -X_Coefficient_Estimation.py"
python "1 qubit- B and C Estimation(3).py"
python "1 qubit_omega_adaptive.py"
python "1 qubit_BC_refinement.py"
```

### Three-qubit workflow

```bash
python "3 qubits-Structure Learning(1).py"
python "3 qubits-B and C estimate(1).py"
python "3 qubit_omega_adaptive(1).py"
python "3 qubit_BC_refinement(1).py"
```

Each script prints its results to the terminal.

These commands illustrate the conceptual learning order. The scripts
are standalone examples and do not automatically transfer results
between separate executions.

## 5. Passing Results Between Stages

The frequency-estimation scripts contain predefined preliminary
coefficient estimates in `LEARNED_B_AND_C`.

The refinement scripts contain predefined upstream results in
`DEFAULT_INPUT` or `DEFAULT_INPUTS`.

To use newly computed results, update these records or pass result
dictionaries directly through the available functions:

| Stage | Single-qubit function | Three-qubit function |
|---|---|---|
| Frequency estimation | `learn_omega_for_single_qubit(bc_result=...)` | `learn_omega_for_all_targets(learned_coefficients=...)` |
| Coefficient refinement | `refine_single_qubit(frequency_result=...)` | `refine_all_targets(frequency_results=...)` |

For three-qubit frequency estimation, organize coefficient records by
Pauli label. Each record should include `B_hat`, `C_hat`, `epsilon_B`,
and `epsilon_C`.

Keep the simulated Hamiltonian, coefficient inputs, frequency priors,
and error bounds consistent when changing the examples.

## 6. Main Settings

Settings are defined in the source files and their `__main__` blocks.

| Setting | Meaning |
|---|---|
| `B_max`, `C_max` | Prior upper bounds on coefficient magnitudes. |
| `omega_min`, `omega_max` | Prior frequency range. |
| `t_short` | Short interrogation time for estimating B + C. |
| `T_total` | Evolution-time setting; its role depends on the learning stage. |
| `shots` | Measurement samples per observable; `None` uses ideal expectations in coefficient estimation and refinement. |
| `failure_probability` | Failure-probability setting for each frequency-learning target. |
| `target_B_error`, `target_C_error` | Requested coefficient-interval accuracy in refinement. |
| `max_B_time`, `max_C_time` | Maximum single interrogation times in refinement. |
| `seed` | Random seed used by stochastic routines. |

## 7. Understanding the Outputs

- `B_hat_fit`: preliminary B estimate from the long-time phase fit.
- `B_plus_C_hat`: short-time estimate of B + C.
- `C_hat`: preliminary C estimate.
- `epsilon_B`, `epsilon_C`: coefficient error bounds under the assumptions used by the corresponding stage.
- `omega_hat`: estimated frequency.
- `frequency_error_bound`: frequency error bound.
- `B_hat_ref`, `C_hat_ref`: refined coefficient estimates.
- `epsilon_B_ref`, `epsilon_C_ref`: maximum distances from the refined estimates to their candidate intervals.
- `absolute_error`: error relative to the simulated true parameter, used for diagnostics.

Frequency-learning outputs also report the sensing time, grid size,
QSP degree, search-query counts, filter validation, and response region.

## 8. Scope and Assumptions

These scripts are numerical simulation examples.

Frequency estimation uses an effective sensing model with ideal
reshaping/scaling and coherent queries. Its guarantees depend on valid
input coefficient intervals and the stated sensing/filter conditions.

Refinement combines prior intervals with measurement and model-error
bounds. A requested target accuracy may remain unattained because of
time limits, frequency uncertainty, measurement uncertainty, or
unresolved phase aliases. Inspect the returned status and intervals.

For three-qubit frequency learning, `failure_probability` is specified
per target term. A joint failure bound can be obtained by summing the
individual failure bounds.

Randomized measurement and search outputs may change when the seed or
settings are changed.
