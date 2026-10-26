# Learning Periodically Time-Dependent Quantum Hamiltonians

Numerical simulations for identifying the Pauli support and estimating the coefficients and driving frequencies of time-dependent Hamiltonians in single-qubit and three-qubit systems.

The implementation combines Bell-basis support learning, target-selective Hamiltonian reshaping, multiscale Ramsey estimation, quantum signal processing (QSP), BBHT amplitude search, and coefficient refinement. Each `run_all.py` computes fresh results and passes them directly to the next stage.

## 1. Hamiltonian Models

### Single qubit

$$
H(t)=AX+\left[B+C\cos(\omega t)\right]Z.
$$

| Parameter | Default value |
|---|---:|
| $A$ | 0.8 |
| $B$ | 0.3 |
| $C$ | 0.1 |
| $\omega$ | 9.8 |

The simulated model is defined by `ModelOracle` in `1- qubit/learning_common.py`. Its expected nonidentity support is `X`, `Z`.

### Three qubits

$$
H(t)=\sum_{s\in S}\left[B_s+C_s\cos(\omega_s t)\right]P_s,
\qquad S=\{\mathrm{XII},\mathrm{IXX},\mathrm{ZXY}\}.
$$

| Pauli label | $B_s$ | $C_s$ | $\omega_s$ |
|---|---:|---:|---:|
| `XII` | 0.720 | 0.080 | 1.0 |
| `IXX` | 0.360 | 0.040 | 1.2 |
| `ZXY` | 0.145 | 0.015 | 1.5 |

These terms are defined in `3-qubits/three_qubit_model.py`. Label characters refer to `q0, q1, q2`, while matrices use the computational-basis order $|q_2q_1q_0\rangle$. Thus `XII` corresponds to $I\otimes I\otimes X$ and `ZXY` to $Y\otimes X\otimes Z$.

The simulations use $\hbar=1$ and angular frequencies in radians per unit time. The three-qubit model has multiple drive frequencies; the routines estimate each term's frequency separately.

## 2. Installation

Use Python **3.10 or newer** to run all examples, including the standalone attenuation scan.

Clone the repository and install the core dependencies:

```bash
git clone https://github.com/Jingyi111777/Learning-code-for-time-periodic-Hamiltonian.git
cd Learning-code-for-time-periodic-Hamiltonian
python -m pip install numpy scipy
```

Install Matplotlib for the optional attenuation-scan plots:

```bash
python -m pip install matplotlib
```

MindQuantum is optional. It is imported only when explicitly calling `execute_six_qubit_bell_measurement(...)` in the three-qubit structure-learning module:

```bash
python -m pip install mindquantum
```

The default pipelines and validation tests use NumPy/SciPy and do not require MindQuantum.

## 3. Quick Start

Run each workflow from its own folder because the two implementations use the same module names. Keep the files in their respective folders.

### Single-qubit pipeline

From the repository root:

```bash
cd "1- qubit"
python run_all.py --output results.json
```

This runs five stages:

1. Identify the nonidentity Pauli support.
2. Estimate the static coefficient $A$.
3. Obtain preliminary estimates and error bounds for $B$ and $C$.
4. Estimate $\omega$ using the ESU sensing response, QSP, and BBHT search.
5. Refine $B$ and $C$ using the frequency estimate and its uncertainty.

### Three-qubit pipeline

From the repository root:

```bash
cd "3-qubits"
python run_all.py --output results.json
```

This runs four stages:

1. Identify the nonidentity Pauli strings.
2. Build target-selective reshaping groups and estimate $B_s$, $C_s$ for each discovered term.
3. Estimate each $\omega_s$ using QSP and BBHT search.
4. Refine $B_s$, $C_s$ for all targets.

Both pipelines print results and write a JSON file. The output path is relative to the current working directory; an existing file at that path is overwritten. Use another filename to preserve an earlier run.

### Command-line options

| Option | Single qubit | Three qubits | Meaning |
|---|---|---|---|
| `--output` | Yes | Yes | JSON output path; default: `results.json`. |
| `--shots` | Yes | Yes | Positive integer: samples per Ramsey quadrature in coefficient estimation and refinement. Omitted: exact expectation values. |
| `--T-B` | No | Yes | Preliminary $B_s$ interrogation-time cap; default: `128.0`. |
| `--t-short` | No | Yes | Short Ramsey time for preliminary $B_s+C_s$ estimation; default: `0.2`. |

For example, from either workflow folder:

```bash
python run_all.py --shots 1000000000 --output results_finite_shots.json
python run_all.py --help
```

`--shots` does not set the Bell-sampling budget for structure learning or replace the coherent-query frequency-search model. Structure learning has its own sampling settings. Too few Ramsey shots can produce uninformative bounds or violate the search-design conditions; the code raises an error in these cases.

## 4. Files and Responsibilities

### `1- qubit/`

| File | Purpose |
|---|---|
| `run_all.py` | Run the full five-stage workflow and serialize fresh results. |
| `learning_common.py` | Define the model, external priors, Ramsey measurements, ESU integrals, phase intervals, and optional finite-pulse Ramsey backend. |
| `structure_learning.py` | Identify Pauli support from random-time-window Bell measurements. |
| `x_coefficient_estimation.py` | Estimate $A$ by multiscale Ramsey confidence-interval updates. |
| `bc_estimation.py` | Estimate $B$ using geometrically doubled Ramsey times and $B+C$ using short-time Richardson extrapolation. |
| `omega_adaptive.py` | Design the frequency grid, synthesize and check the QSP filter, and run BBHT search. |
| `bc_refinement.py` | Refine $B$, $C$ while retaining measurement, frequency, and phase-alias uncertainty. |
| `test_validation.py` | Scientific regression tests. |

### `3-qubits/`

| File | Purpose |
|---|---|
| `run_all.py` | Run the full four-stage workflow and serialize fresh results. |
| `three_qubit_model.py` | Define the three-term model, Pauli conventions, reshaping groups, generalized Ramsey probes, and optional finite-pulse backend. |
| `numerical_utils.py` | Shared shot validation, phase-error bounds, ESU integrals, and interval operations. |
| `structure_learning.py` | Recover support using 8D propagation and the equivalent 64-outcome six-qubit Bell distribution; includes an optional MindQuantum circuit. |
| `estimation_core.py` | Implement preliminary Ramsey coefficient estimation for one target. |
| `bc_estimation.py` | Estimate coefficients for the discovered targets and record their reshaping groups. |
| `omega_adaptive.py` | Estimate the frequency of each target with QSP/BBHT. |
| `bc_refinement.py` | Refine all targets' static coefficients and modulation amplitudes. |
| `lambda_three_values_standalone.py` | Independently compare attenuation values using repeated frequency searches and coefficient refinement. |
| `test_validation.py` | Scientific regression tests. |
| `results.json` | Saved pipeline output; rerun `run_all.py` to generate fresh results. |
| `lambda_three_results/` | Saved attenuation-scan tables, metadata, and figures. |

The standalone attenuation scan is optional and is not called by `run_all.py`.

## 5. Running Individual Stages

From `1- qubit/`:

```bash
python structure_learning.py
python x_coefficient_estimation.py
python bc_estimation.py
python omega_adaptive.py
python bc_refinement.py
```

From `3-qubits/`:

```bash
python structure_learning.py
python bc_estimation.py
python omega_adaptive.py
python bc_refinement.py
```

These commands print individual-stage results. They do not read the previous command's terminal output or automatically load `results.json`. Frequency estimation and refinement compute upstream results when inputs are omitted. Use `run_all.py` to execute one consistent chain, or pass dictionaries through the Python functions below.

| Stage | Single-qubit API | Three-qubit API |
|---|---|---|
| Support identification | `learn_structure(...)` | `learn_structure(...)` |
| Static $X$ coefficient | `learn_x_coefficient(...)` | Not a separate stage. |
| Preliminary coefficients | `learn_BC(...)` | `learn_B_and_C_for_all_targets(support, ...)` |
| Frequency estimation | `learn_omega_for_single_qubit(bc_result, ...)` | `learn_omega_for_all_targets(coefficients, ...)` |
| Refinement | `refine_single_qubit(frequency_result, config)` | `refine_all_targets(frequency_results, config, ...)` |

## 6. Attenuation Scan

The standalone three-qubit experiment compares how attenuation $\lambda$ affects frequency accuracy, refined coefficients, and sensing resources. It reuses preliminary coefficient estimates across repeated trials.

From `3-qubits/`:

```bash
python lambda_three_values_standalone.py --target IXX --lambdas 0.6 0.2 0.05 --repeats 30 --output lambda_three_results
```

The default target is `IXX`; supported targets are `XII`, `IXX`, and `ZXY`. The default attenuation values are `0.6`, `0.2`, and `0.05`, with 30 trials per value. For a shorter initial run, use `--repeats 1`.

The scan uses refinement defaults of `--target-B-error 1e-6`, `--target-C-error 1e-6`, `--max-B-time 1024`, and `--max-C-time 256`. These differ from the modular pipeline's refinement defaults.

| Output | Contents |
|---|---|
| `trials.csv` | Per-trial estimates, error bounds, resource counts, and status. |
| `summary.csv` | Per-attenuation summaries, including successful and failed trial counts. |
| `details.json` | Preliminary estimates, filter designs, settings, and summary metadata. |
| `lambda_accuracy_<target>.png/.pdf/.svg` | Actual estimation errors and reported bounds. |
| `lambda_resources_<target>.png/.pdf/.svg` | ESU interrogation time, candidate count, and search queries. |

Plots are generated when at least one trial succeeds. Means and quantiles use successful trials only; inspect failure counts as well. Individual-run bounds are not simultaneous confidence intervals for all scan rows.

## 7. Understanding the Results

| Field | Meaning |
|---|---|
| `structure.support` | Identified nonidentity Pauli labels. |
| `theorem_sample_budget_met` | Whether the configured structure-learning sample budget meets the implemented requirement. |
| `A_hat`, `error_bound` | Estimated static $X$ coefficient and interval radius. |
| `B_hat`, `C_hat` | Preliminary coefficient estimates. |
| `epsilon_B`, `epsilon_C` | Preliminary coefficient error bounds. |
| `omega_hat`, `frequency_error_bound` | Estimated angular frequency and its reported error bound. |
| `filter_check` | QSP filter check and numerical certificate information. |
| `B_hat_ref`, `C_hat_ref` | Refined coefficient estimates. |
| `epsilon_B_ref`, `epsilon_C_ref` | Refined interval radii. |
| `intervals` | Remaining candidate intervals, including separate phase aliases where present. |
| `status` | Whether the requested refinement accuracy was attained. |
| `union_failure_probability_bound` | Pipeline failure bound assembled from the stage budgets, conditional on the stated assumptions. |

The single-qubit JSON contains `structure`, `X`, `BC`, `frequency`, and `refinement`. The three-qubit JSON contains `structure`, `coefficients`, `frequency`, and `refinement`, with target-dependent records keyed by Pauli label.

Refinement statuses include:

| Status | Interpretation |
|---|---|
| `target_reached` | The returned error radius meets the requested target. |
| `time_or_frequency_limit` | The available schedule and uncertainty bounds did not reach the target. |
| `no_admissible_period` | No allowed interrogation period was available. |
| `unresolved_phase_aliases` | Multiple phase-compatible candidate intervals remain. |

A successful program exit does not imply every requested refinement target was reached. Inspect both the status and the returned intervals.

## 8. Validation

Run the existing tests separately in each workflow folder:

```bash
python -m unittest -v test_validation.py
```

Tests cover ESU integrals, Ramsey phase conventions, signed coefficients, interval bounds and aliases, support recovery, QSP derivatives and filter checks, BBHT/refinement integration, and optional finite-pulse Ramsey calculations. The three-qubit tests also check matrix order, Bell-label mapping, six-qubit Bell probabilities, and target-selective reshaping.

## 9. Configuration and Simulation Scope

- **Models and priors:** Change `ModelOracle`/`Priors` for the single-qubit example, or `TERMS`/`EstimationPriors` for three qubits. Keep structure-learning bounds, coefficient priors, frequency ranges, and any standalone-scan model consistent with those changes.
- **Refinement:** Configure `bc_refinement.Config` through Python for target errors and interrogation-time limits. The modular defaults are $10^{-4}$ for both coefficient targets, `max_B_time=1024`, and `max_C_time=64`. These are maximum single-interrogation times, not total experimental time.
- **Measurement model:** Omitted `--shots` means infinite-shot Ramsey expectations. It does not represent a finite experimental resource cost. Structure learning still samples Bell outcomes, and frequency search remains stochastic.
- **Ideal control:** Default coefficient and frequency routines assume ideal term isolation/reshaping, attenuation, and coherent ESU queries. Optional `PulseRamseyOracle` implementations model finite deterministic reshaping for ordinary Ramsey measurements; they do not implement a full finite-pulse coherent QSP pipeline.
- **Certificates and bounds:** Error and failure bounds depend on valid external priors, the model and control assumptions, and QSP checks performed in floating-point arithmetic. Search budgets can be exhausted and design conditions can fail explicitly.
- **Reference truth:** Simulated true parameters belong to the model source and are also used in diagnostic coverage/error checks; estimators use measurements and supplied priors.
- **Changing support:** The single-qubit runner explicitly requires support `X`, `Z`. The three-qubit runner uses discovered support, subject to the assumptions and reshaping conditions implemented by the routines.
- **Reproducibility:** Random seeds are defined in the functions/configuration. Results may change with seeds, settings, and numerical-library versions. The repository does not pin dependency versions.
