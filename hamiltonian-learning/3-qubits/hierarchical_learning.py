"""Amplitude-layered support recovery with estimated Hamiltonian compensation."""
import numpy as np
from scipy.stats import binom

from bc_estimation import learn_B_and_C_for_all_targets
from structure_learning import DEFAULT_GAMMA_BOUND, compensation_norm_bounds, learn_structure
from three_qubit_model import LABELS, PeriodicPauliTerm


DEFAULT_THRESHOLDS = (.60, .30, .10)
DEFAULT_HIERARCHICAL_BELL_SHOTS = 10**10
VALIDATION_FAILURE_BUDGET = .02


def validate_candidates(candidates, measurements, known_terms, failure_budget):
    """Test each proposed label against the absent-label Bell probability bound."""
    t, r = measurements['t'], measurements['r']
    norm, derivative, commutator = compensation_norm_bounds(known_terms, DEFAULT_GAMMA_BOUND)
    # First-order Pauli coefficient is zero for an absent, uncompensated label.
    # Add the propagation discretization bound to the short-time Dyson remainder.
    amplitude_bound = ((norm**2 + (derivative + commutator) / r) * t**2 / 2
                       + measurements['numerical_amplitude_error_bound'])
    null_probability = min(1., amplitude_bound**2)
    shots = measurements['shots_per_window']
    cutoff = failure_budget / (len(candidates) * measurements['windows'])
    checks = {}
    for label in candidates:
        index = LABELS.index(label)
        counts = np.array([window['counts'][index] for window in measurements['records']])
        best = int(counts.argmax())
        p_value = float(binom.sf(counts[best] - 1, shots, null_probability))
        checks[label] = dict(
            accepted=p_value <= cutoff, min_window_p_value=p_value, cutoff=cutoff,
            max_count=int(counts[best]), window_start=measurements['records'][best]['t0'],
        )
    return null_probability, checks


def learn_hierarchically(source, priors, finish_parameters, *, thresholds, bell_shots, seed,
                         shots, T_B, t_short):
    thresholds = tuple(float(value) for value in thresholds)
    if (len(thresholds) < 2 or not np.isfinite(thresholds).all()
            or thresholds[-1] <= 0
            or any(a <= b for a, b in zip(thresholds, thresholds[1:]))):
        raise ValueError('Hierarchical thresholds must strictly decrease to a positive a_min.')
    support = set()
    known_terms = ()
    rounds = []
    coefficients, term_types, frequency, refined = {}, {}, {}, {}
    validation_budget = VALIDATION_FAILURE_BUDGET / len(thresholds)
    seeds = np.random.SeedSequence(seed).spawn(len(thresholds))
    for index, (a, round_seed) in enumerate(zip(thresholds, seeds), start=1):
        proposal_seed, validation_seed, coefficient_seed, search_seed, refinement_seed = (
            int(value) for value in round_seed.generate_state(5, dtype=np.uint64)
        )
        options = dict(source=source, a_min=a, known_terms=known_terms,
                       shots_per_window=bell_shots, omega_min=priors.omega_min,
                       omega_max=priors.omega_max)
        proposal = learn_structure(seed=proposal_seed, **options)
        candidates = sorted(set(proposal['support']) - support)
        validation_shots = 0
        null_probability, checks = None, {}
        if candidates:
            validation = learn_structure(seed=validation_seed, **options)
            validation_shots = validation['total_bell_samples']
            null_probability, checks = validate_candidates(
                candidates, validation, known_terms, validation_budget,
            )
        confirmed = sorted(label for label, check in checks.items() if check['accepted'])
        final_round = index == len(thresholds)
        estimates = {}
        if confirmed:
            estimates = learn_B_and_C_for_all_targets(
                sorted(support.union(confirmed)), source, priors,
                T_B=T_B, t_short=t_short, shots=shots, seed=coefficient_seed,
                complete_reshaping=True,
            )
        amplitude_lower_bounds = {
            label: max(0., abs(estimates[label]['B_hat']) - estimates[label]['epsilon_B'],
                       abs(estimates[label]['C_hat']) - estimates[label]['epsilon_C'])
            for label in confirmed
        }
        # The final round uses a_min for the proposal design and admits any independently
        # confirmed term, avoiding an extra confidence-bound cutoff exactly at a_min.
        new_terms = [
            label for label in confirmed if final_round or amplitude_lower_bounds[label] >= a
        ]
        observation_end = 2 * np.pi / priors.omega_min + proposal['t']
        compensation_error_bound = sum(
            refined[term.label]['epsilon_B_ref'] + refined[term.label]['epsilon_C_ref']
            for term in known_terms
        ) + sum(
            abs(term.C) * min(2., observation_end * frequency[term.label]['frequency_error_bound'])
            for term in known_terms if term.omega is not None
        )
        stage = dict(
            round=index, a_j=a, final_round=final_round, compensation_in=known_terms,
            compensation_error_bound=compensation_error_bound,
            proposal_seed=proposal_seed, validation_seed=validation_seed,
            coefficient_seed=coefficient_seed, search_seed=search_seed,
            refinement_seed=refinement_seed, t=proposal['t'], r=proposal['r'],
            windows=proposal['windows'], shots_per_window=proposal['shots_per_window'],
            proposal_bell_samples=proposal['total_bell_samples'],
            validation_bell_samples=validation_shots,
            validation_failure_budget=validation_budget,
            absent_label_probability_bound=null_probability,
            proposed=candidates, candidate_checks=checks, confirmed=confirmed,
            amplitude_lower_bounds=amplitude_lower_bounds, new_terms=new_terms,
            esu_queries=0,
        )
        if new_terms:
            support.update(new_terms)
            coefficients = {label: estimates[label] for label in sorted(support)}
            term_types, frequency, refined = finish_parameters(
                coefficients, search_seed, refinement_seed,
            )
            known_terms = tuple(
                PeriodicPauliTerm(label, row['B_hat_ref'], row['C_hat_ref'],
                                  frequency[label]['omega_hat'] if label in frequency else None)
                for label, row in refined.items()
            )
            stage['esu_queries'] = sum(record['esu_queries'] for record in frequency.values())
        stage.update(support=sorted(support), compensation_out=known_terms)
        rounds.append(stage)

    # Simulator truth is used only after all support updates for scoring.
    truth = set(source.by_label)
    for stage in rounds:
        stage['false_positives'] = sorted(set(stage['new_terms']) - truth)
    structure = dict(
        mode='hierarchical', support=sorted(support), learned_support=sorted(support),
        a_min=thresholds[-1], thresholds=thresholds, seed=seed, rounds=rounds,
        total_bell_samples=sum(row['proposal_bell_samples'] + row['validation_bell_samples']
                               for row in rounds),
        total_esu_queries=sum(row['esu_queries'] for row in rounds),
        validation_failure_budget=VALIDATION_FAILURE_BUDGET,
        validation_scope='false-positive checks conditional on the short-time bounds',
        failure_probability=None,
        diagnostics=dict(false_positives=sorted(support - truth), missed_terms=sorted(truth - support)),
    )
    return structure, coefficients, term_types, frequency, refined
