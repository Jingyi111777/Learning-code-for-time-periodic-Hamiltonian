# Learning the Structure and Parameters of Periodically Modulated Hamiltonians

This repository accompanies the manuscript by Chunxia Li, Changhao Yi, and Yongping Zhang. It contains the single-qubit and three-qubit simulations from the *Numerical examples* section. The simulations recover Pauli support from short-time Bell sampling, estimate coefficients with Ramsey measurements, locate modulation frequencies with QSP–BBHT search, and refine the coefficient estimates using the recovered frequencies.

## Setup

Extract the archive and use the directory containing this README as the project root.
It contains `1-qubit/` and `3-qubits/`. 
The reference run used Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0,
Matplotlib 3.10.8, and Pillow 13.0.0.

```bash
python -m pip install -r requirements.txt
```

Matplotlib is used for the paper figures. MindQuantum is optional and is used
only by the separate six-qubit Bell-circuit helper.

## Reproduce Figures 2–4

Run these commands from the project root. The first uses the supplied result
JSON files. The second regenerates all four simulation inputs and then plots them.

```bash
python reproduce_figures.py --output-dir figures
python reproduce_figures.py --rerun --output-dir reproduced_figures
```

Both commands produce the PNG filenames used in the manuscript and
`figure_manifest.json`, which records the input and output SHA-256 hashes,
code version, plotting environment, and font. The rerun saves its JSON files
under `reproduced_figures/simulation_results/` and starts each simulation with
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. The plots use unrounded
estimates. Figure 2 uses the exact-expectation single-qubit run. Figure 3
uses the detuned three-qubit baseline. Figure 4 uses that baseline and
the two finite-shot three-qubit runs. Liberation Serif matches the paper
typography. The script reports a DejaVu Serif fallback if needed.

The archive identifies this source snapshot as `2026.11.14-f05.3` in
`VERSION`. The `release_manifest.json` file records the SHA-256 of each
packaged file and a combined SHA-256 for the Python sources. Refer to this
snapshot identifier when citing the bundled simulations. Fresh main-run
JSON files record the runtime and random seeds in `run_metadata`. The
frequency records provide the bin count, interrogation time, outer threshold,
QSP degree, attenuation, search seeds, and success flag. The earlier
commensurate-frequency reference files are archived historical outputs.
The packaged simulation results retain the versions of their original runs.

## Single-qubit example

The simulator uses $H_1(t)=AX+[B+C\cos(\omega t)]Z$ with $(A,B,C,\omega)=(0.8,0.3,0.1,9.8)$. The baseline uses 35 Bell-sampling windows with $10^7$ shots per window.

The single-qubit model uses `typing.Optional[float]` for the frequency so
the annotation can also be evaluated by Python versions before 3.10.
For the pinned reproduction environment above, use Python 3.12.

From the repository root, run the baseline and the finite-shot example:

```bash
cd "1- qubit"
python run_all.py --output reproduced_single_baseline.json
python run_all.py --shots 1000000000 --output reproduced_single_1e9.json
```

The archive includes the baseline as `1- qubit/results.json`. It recovers the support $\{X,Z\}$ and returns $\widehat{\omega}\approx9.73$ after 108 ESU queries.

For a static $Z$ term, run this command from `1- qubit/`:

```bash
python run_all.py --c 0 --output reproduced_single_static.json
```

The classifier uses the preliminary Ramsey interval and the prior $C=0$
or $|C|\ge C_{\min}$, with `--c-min` defaulting to 0.01. When only the
static case is compatible, the workflow refines $B$ with ordinary
Ramsey measurements and records $\widehat C=0$, `omega_hat: null`,
`status: "not_applicable"`, and zero ESU queries. If the interval admits
both cases, the run reports that more precise Ramsey estimates are needed.
Add `--shots 1000000000` to run the static example with finite Ramsey shots.

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

The baseline uses the default `single-pass` mode. It starts from an empty support and uses one support-recovery pass with 35 Bell-sampling windows and $2\times10^8$ shots per window. From the repository root, run:

```bash
cd "3-qubits"
python run_all.py --frequency-case detuned --output reproduced_detuned.json
python run_all.py --frequency-case detuned --shots 1000000000 --output reproduced_detuned_1e9.json
python run_all.py --frequency-case detuned --shots 10000000000 --output reproduced_detuned_1e10.json
```

The supplied files `results_detuned.json`, `results_detuned_1e9.json`, and `results_detuned_1e10.json` correspond to these three runs. The baseline uses 216, 204, and 68 ESU queries for XII, IXX, and ZXY, respectively. The archive also retains an earlier commensurate-frequency run in `results.json`.

To reproduce the three-qubit Bell-budget comparison, run the scan from
`3-qubits/` with one numerical-library thread. On Linux or macOS, run:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python structure_learning.py --frequency-case detuned --scan-shots 200000000 --trials 500 --seed 1234 --output reproduced_bell_budget_detuned.json
```

On Windows PowerShell, run:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
python structure_learning.py --frequency-case detuned --scan-shots 200000000 --trials 500 --seed 1234 --output reproduced_bell_budget_detuned.json
```

The supplied `bell_budget_detuned.json` records the reference run on Linux
with Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and both thread settings
set to 1. All 500 trials recovered the exact support at
$2\times10^8$ shots per window. The scan reports the empirical statistics
for both budgets and saves the model, seed, runtime versions, platform,
and thread settings when `--output` is supplied.

### Hierarchical support recovery

Run the three-round numerical variant of Algorithm 1 and its Monte Carlo
support check from `3-qubits/`.

```bash
python run_all.py --structure-mode hierarchical --amplitude-thresholds 0.60 0.30 0.10
python benchmark_hierarchy.py --trials 50 --first-seed 2000
```

Each round proposes Pauli labels from compensated Bell data and checks them on an independent set of Bell windows. The check uses a binomial tail under the short-time upper bound for an absent label, with a Bonferroni allocation across candidates and windows. In intermediate rounds, a verified candidate joins the support when the preliminary Ramsey lower bound on $\max(|B_s|,|C_s|)$ reaches the current threshold. The last threshold sets $a_{\min}$ for the Bell design. The final round admits independently verified candidates, including terms near the amplitude boundary.

After each support update, the code estimates the parameters of the accumulated support and uses those estimates to compensate the next round. Ideal Pauli averaging for a candidate uses all Pauli operators commuting with that target and so can isolate it before the full support is known. The default run adds XII, IXX, and ZXY in that order. The order comes from amplitude thresholds and can change with the model. Candidates may be proposed and confirmed before the round in which they enter the support.

The default hierarchical budget is $10^{10}$ Bell shots per window in each proposal or validation pass, for $2.10\times10^{12}$ Bell samples across the three rounds of the supplied run. The results and round diagnostics are in `results_detuned_hierarchical.json` and `run_all_hierarchical.log`. In the 50 trials with seeds 2000–2049, each run added the same terms in the same order and recovered the exact support. The command reproduces these empirical results in `hierarchy_benchmark.json`. The candidate checks control false positives conditional on the short-time bounds. Full hierarchical recovery is assessed empirically under ideal controls.

## Reproduce Figure 5

From `3-qubits/`, run:

```bash
python lambda_three_values_standalone.py --frequency-case detuned --target IXX --lambdas 0.60 0.20 0.05 --repeats 10 --seed 42 --output reproduced_attenuation
```

The script holds the preliminary coefficient estimates fixed and runs 10 searches per attenuation factor, with trial seeds 42–51 and a design failure probability of 0.01 per BBHT search. It writes `details.json`, `summary.csv`, `trials.csv`, and plots to the selected output folder. The supplied detuned run is in `lambda_three_results_detuned/`. The earlier comparison is in `lambda_three_results/`.

A trial counts as successful when the returned frequency lies inside its
designed error bound and coefficient refinement completes. BBHT timeouts,
out-of-bound estimates, and refinement errors are recorded as failures in
`trials.csv` with their failure reasons. The summary statistics use the
successful trials. All 30 trials in the supplied detuned scan succeeded.

The command generates the single-column, 2-by-2 Figure 5 as
`three_qubit_attenuation_bars.png`. Panel (b) shows the mean absolute
relative errors of the refined coefficients $B$ and $C$ in units of
$10^{-2}\%$. The `B_relative_error_mean` and `C_relative_error_mean` columns
in `summary.csv` store percentages, computed by averaging
$100\,|\widehat{\theta}-\theta|/|\theta|$ over successful trials.
For example, the mean error $0.1586\%$ appears as $15.9$ after scaling
and rounding to three significant digits. The table below reports
the means in percent.

| Attenuation factor | Mean relative error of $B$ | Mean relative error of $C$ |
|---:|---:|---:|
| 0.60 | 0.004392% | 0.1586% |
| 0.20 | 0.001205% | 0.07933% |
| 0.05 | 0.006591% | 0.04139% |

## Options and simulation scope

- Omitting `--shots` uses exact Ramsey expectation values. With `--shots N`, the simulation draws binomial counts for each Ramsey quadrature using `N` shots. Bell counts are sampled separately from multinomial distributions.
- On either `run_all.py`, `--bell-shots-per-window N` sets the Bell sample count. `--theoretical-bell-budget` selects the single-pass reference count for each sampling pass. The reference count uses a target support-recovery failure probability of 0.02 under the single-pass theorem's hypotheses. The support checks report observed recovery at the lower Bell budgets, for which `theorem_sample_budget_met` is `false`.
- `--output` on either `run_all.py` selects the JSON filename. The JSON includes the recovered support, fitted coefficients and frequencies, refinement bounds, and simulation diagnostics.
- The examples assume ideal Hamiltonian reshaping and coherent ESU queries. Attenuation rescales the signal. Finite Ramsey shots model measurement sampling noise, while cumulative errors from finite control sequences are addressed analytically in the manuscript.

Run the validation suite from either extracted example folder:

```bash
python -m unittest test_validation
```

From the project root, verify the plotted statistics against the saved results:

```bash
python -m unittest test_paper_data
```

The suites cover the supplied oracle, the single- and three-qubit static-term branches,
finite reshaping with incomplete support, and BBHT budgets for
$K=64,128,512$ and $\delta=0.01,0.05$.
