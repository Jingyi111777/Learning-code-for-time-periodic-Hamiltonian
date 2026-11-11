# Learning the Structure and Parameters of Periodically Modulated Hamiltonians

This repository accompanies the manuscript by Chunxia Li, Changhao Yi, and Yongping Zhang. It contains the single-qubit and three-qubit simulations from the *Numerical examples* section. The simulations recover Pauli support from short-time Bell sampling, estimate coefficients with Ramsey measurements, locate modulation frequencies with QSP–BBHT search, and refine the coefficient estimates using the recovered frequencies.

## Setup

Extract both archives into the same directory. Place this README beside the extracted `1- qubit/` and `3-qubits/` folders. The space in `1- qubit` is part of the folder name.

The simulations were tested with Python 3.12, NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8.

```bash
python -m pip install numpy==2.3.5 scipy==1.17.0 matplotlib==3.10.8
```

Matplotlib is used by the attenuation plotting script. MindQuantum is optional and is used only by the separate six-qubit Bell-circuit helper.

## Single-qubit example

The simulator uses $H_1(t)=AX+[B+C\cos(\omega t)]Z$ with $(A,B,C,\omega)=(0.8,0.3,0.1,9.8)$. The baseline uses 35 Bell-sampling windows with $10^7$ shots per window.

From the repository root, run the baseline and the finite-shot example:

```bash
cd "1- qubit"
python run_all.py --output reproduced_single_baseline.json
python run_all.py --shots 1000000000 --output reproduced_single_1e9.json
```

The archive includes the baseline as `1- qubit/results.json`. It recovers the support $\{X,Z\}$ and returns $\widehat{\omega}\approx9.73$ after 108 ESU queries.

To repeat the 500-trial support check from the same folder, run:

```bash
python -c "import numpy as np; from structure_learning import learn_structure; seeds = np.random.default_rng(1234).integers(0, np.iinfo(np.int64).max, size=500); print(sum(learn_structure(seed=int(s), shots_per_window=10000000)['support'] == ['X', 'Z'] for s in seeds))"
```

The command prints the number of exact support recoveries, `500` in the tested environment.

## Three-qubit example

The default `detuned` model uses the terms below. Each Pauli string has a time-dependent coefficient $B_s+C_s\cos(\omega_s t)$.

| Pauli string | $B_s$ | $C_s$ | $\omega_s$ |
|---|---:|---:|---:|
| XII | 0.7200 | 0.08000 | 1.037 |
| IXX | 0.3600 | 0.04000 | 1.217 |
| ZXY | 0.1450 | 0.01500 | 1.463 |

The baseline uses 35 Bell-sampling windows with $2\times10^8$ shots per window. From the repository root, run:

```bash
cd "3-qubits"
python run_all.py --frequency-case detuned --output reproduced_detuned.json
python run_all.py --frequency-case detuned --shots 1000000000 --output reproduced_detuned_1e9.json
python run_all.py --frequency-case detuned --shots 10000000000 --output reproduced_detuned_1e10.json
```

The supplied files `results_detuned.json`, `results_detuned_1e9.json`, and `results_detuned_1e10.json` correspond to these three runs. The baseline uses 216, 204, and 68 ESU queries for XII, IXX, and ZXY, respectively. The archive also retains an earlier commensurate-frequency run in `results.json`.

To repeat the three-qubit Bell-budget comparison, run the following from `3-qubits/`:

```bash
python structure_learning.py --frequency-case detuned --scan-shots 200000000 --trials 500 --seed 1234
```

The script prints the empirical support-recovery statistics for both the chosen shot count and the theoretical reference. The lower-budget run recovered the exact support in 499 of 500 trials. The supplied `bell_budget_detuned.json` records the comparison.

## Attenuation scan

From `3-qubits/`, run:

```bash
python lambda_three_values_standalone.py --frequency-case detuned --target IXX --lambdas 0.60 0.20 0.05 --repeats 10 --seed 42 --output reproduced_attenuation
```

The script holds the preliminary coefficient estimates fixed and runs 10 searches per attenuation factor, with trial seeds 42–51 and a design failure probability of 0.01 per BBHT search. It writes `details.json`, `summary.csv`, `trials.csv`, and plots to the selected output folder. The supplied detuned run is in `lambda_three_results_detuned/`. The earlier comparison is in `lambda_three_results/`.

The plotting script in this archive displays mean refined interval radii in panel (b) of `three_qubit_attenuation_bars.png`. For the manuscript's mean relative coefficient errors, use `B_error`, `C_error`, `B_true`, and `C_true` from `trials.csv`. Compute $100|\widehat{\theta}-\theta|/|\theta|$ for each trial and average the percentages by attenuation factor. The supplied data give:

| Attenuation factor | Mean relative error of $B$ | Mean relative error of $C$ |
|---:|---:|---:|
| 0.60 | 0.004392% | 0.1586% |
| 0.20 | 0.001205% | 0.07933% |
| 0.05 | 0.006591% | 0.04139% |

## Options and simulation scope

- Omitting `--shots` uses exact Ramsey expectation values. With `--shots N`, the simulation draws binomial counts for each Ramsey quadrature using `N` shots. Bell counts are sampled separately from multinomial distributions.
- On either `run_all.py`, `--bell-shots-per-window N` sets the Bell sample count. `--theoretical-bell-budget` selects the theoretically sufficient count. The theoretical comparison uses a target support-recovery failure probability of 0.02. The support checks report observed recovery at the lower Bell budgets, for which `theorem_sample_budget_met` is `false`.
- `--output` on either `run_all.py` selects the JSON filename. The JSON includes the recovered support, fitted coefficients and frequencies, refinement bounds, and simulation diagnostics.
- The examples assume ideal Hamiltonian reshaping and coherent ESU queries. Attenuation rescales the signal. Finite Ramsey shots model measurement sampling noise, while cumulative errors from finite control sequences are addressed analytically in the manuscript.

Run the validation suite from either extracted example folder:

```bash
python -m unittest test_validation
```
