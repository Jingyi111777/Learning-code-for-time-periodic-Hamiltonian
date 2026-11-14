"""Scientific regression tests. Run from this folder: python -m unittest -v."""
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.linalg import expm
from three_qubit_model import (TERMS,PAULIS,LABELS,I8,X,Y,Z,I2,ThreeQubitOracle,
    PeriodicPauliTerm,EstimationPriors,commutes,complete_reshaping_group,reshaping_group,
    ramsey_probe,PulseRamseyOracle)
from structure_learning import learn_structure,bell_probabilities,six_bitstring_to_pauli_label
from bc_estimation import learn_B_and_C_for_all_targets,learn_B_and_C_for_target
from omega_adaptive import (learn_omega_for_all_targets,choose_search_design,
    build_qsp_oracle_blocks,bbht_amplitude_search,_qsp_fit_residual_jacobian,
    bbht_failure_budget,budgeted_bbht_search)
from bc_refinement import refine_all_targets,Config,normalize_input,systematic_phase_bound
from numerical_utils import square_integrals,phase_error
from run_all import classify_coefficient, run
from hierarchical_learning import validate_candidates
import lambda_three_values_standalone as attenuation

SUPPORT=('XII','IXX','ZXY')

class Validation(unittest.TestCase):
    def test_bbht_budget_across_grid_sizes_and_targets(self):
        for size in (64, 128, 512):
            caps = []
            for delta in (.05, .01):
                with self.subTest(size=size, delta=delta):
                    design = SimpleNamespace(number_of_bins=size, failure_probability=delta,
                                             filter_epsilon=delta/(2*size))
                    budget = bbht_failure_budget(design)
                    self.assertAlmostEqual(budget['delta_filter']+budget['delta_search'], delta)
                    self.assertLessEqual(budget['qsp_false_accept_probability_bound'], budget['delta_filter'])
                    self.assertLessEqual(budget['bbht_miss_probability_bound'], budget['delta_search'])
                    blocks = np.repeat(np.eye(2)[None, :, :], size, axis=0)
                    blocks[0] = [[0., -1.], [1., 0.]]
                    result = budgeted_bbht_search(blocks, 3, budget, seed=42)
                    self.assertEqual(result['k_hat'], 0)
                    self.assertLessEqual(result['bbht_rounds'], budget['max_bbht_rounds'])
                    self.assertEqual(result['esu_queries'], 3*result['total_oracle_calls'])
                    caps.append(budget['max_bbht_rounds'])
            self.assertGreater(caps[1], caps[0])

    def test_incomplete_support_leaves_a_finite_reshaping_bias(self):
        source = ThreeQubitOracle.from_frequency_case('detuned')
        time = .2
        rng = np.random.default_rng(2)
        ideal = source.ramsey('ZXY', time, None, rng)
        correct = PulseRamseyOracle(source, SUPPORT, quadrature_bias=.0005)
        incomplete = PulseRamseyOracle(source, ('IXX', 'ZXY'), quadrature_bias=.0005)
        full = correct.ramsey('ZXY', time, None, rng)
        missed = incomplete.ramsey('ZXY', time, None, rng)
        full_error = max(abs(full[q]-ideal[q]) for q in ('R', 'S'))
        missing_error = max(abs(missed[q]-ideal[q]) for q in ('R', 'S'))
        self.assertLessEqual(full_error, full['quadrature_bias_bound'])
        self.assertGreater(missing_error, missed['quadrature_bias_bound'])

    def test_attenuation_summary_excludes_failed_trials(self):
        in_range = dict(k_hat=27, esu_queries=9, total_oracle_calls=9, bbht_rounds=2)
        out_of_range = dict(in_range, k_hat=0)
        timeout = attenuation.BBHTSearchTimeout(4, 9, 1)
        fit = dict(B_hat_ref=.361, C_hat_ref=.04, epsilon_B_ref=.01, epsilon_C_ref=.01,
                   B=dict(status='ok', max_interrogation_time=1.),
                   C=dict(status='ok', max_interrogation_time=1.))
        with (
            TemporaryDirectory() as directory,
            patch.object(attenuation, 'synthesize_qsp_phases',
                         return_value=(np.zeros(2), {'continuous_error_bound': 0.})),
            patch.object(attenuation, 'budgeted_bbht_search',
                         side_effect=[out_of_range, in_range, timeout, in_range]),
            patch.object(attenuation, 'refine_all_targets',
                         side_effect=[{'IXX': fit}, RuntimeError('injected refinement failure')]) as refine,
            patch.object(attenuation, 'make_paper_bars'),
            patch.object(attenuation, 'make_plots'),
        ):
            report, rows = attenuation.scan(lambdas=[.6], repeats=4, output=directory, verbose=False)

        self.assertEqual(refine.call_count, 2)
        self.assertEqual([row['success'] for row in rows], [False, True, False, False])
        self.assertEqual([row['status'] for row in rows],
                         ['trial_failed', 'ok', 'trial_failed', 'trial_failed'])
        self.assertEqual([row['failure_reason'] for row in rows],
                         ['outside_frequency_bound', None, 'bbht_round_limit', 'refinement_error'])
        self.assertGreater(rows[0]['omega_error'], rows[0]['omega_bound'])
        summary = report['summary'][0]
        self.assertEqual((summary['successful_trials'], summary['failed_trials']), (1, 3))
        self.assertAlmostEqual(summary['B_relative_error_mean'], 100 * .001 / .36)
        self.assertAlmostEqual(summary['omega_error_mean'], rows[1]['omega_error'])

    def test_hierarchical_pipeline_passes_estimates_to_next_round(self):
        result = run(structure_mode='hierarchical', verbose=False)
        structure = result['structure']
        rounds = structure['rounds']
        self.assertEqual([row['new_terms'] for row in rounds], [['XII'], ['IXX'], ['ZXY']])
        self.assertEqual(rounds[-1]['a_j'], .10)
        self.assertTrue(rounds[-1]['final_round'])
        self.assertEqual(rounds[0]['compensation_in'], ())
        for previous, current in zip(rounds, rounds[1:]):
            self.assertEqual(current['compensation_in'], previous['compensation_out'])
            self.assertGreater(current['compensation_error_bound'], 0)
        for row in rounds:
            self.assertNotEqual(row['proposal_seed'], row['validation_seed'])
        self.assertEqual(structure['diagnostics'], {'false_positives': [], 'missed_terms': []})
        self.assertIsNone(result['union_failure_probability_bound'])

    def test_hierarchical_final_round_keeps_boundary_term(self):
        source = ThreeQubitOracle((PeriodicPauliTerm('ZZI', .1, 0.),))
        result = run(source=source, structure_mode='hierarchical', verbose=False)
        self.assertEqual([row['new_terms'] for row in result['structure']['rounds']],
                         [[], [], ['ZZI']])
        self.assertEqual(result['frequency']['ZZI']['esu_queries'], 0)

    def test_candidate_validation_rejects_bell_count_below_cutoff(self):
        counts = np.zeros(len(LABELS), dtype=np.int64)
        counts[LABELS.index('XII')] = 40
        counts[LABELS.index('IXX')] = 1
        measurement = dict(t=.001, r=1, windows=1, shots_per_window=10**11,
                           numerical_amplitude_error_bound=0., records=[dict(t0=.5, counts=counts)])
        _, checks = validate_candidates(('XII', 'IXX'), measurement, (), .01)
        self.assertTrue(checks['XII']['accepted'])
        self.assertFalse(checks['IXX']['accepted'])

    def test_complete_reshaping_isolates_target_from_all_labels(self):
        for target in SUPPORT:
            group = complete_reshaping_group(target)
            self.assertEqual(len(group), 32)
            for label in LABELS[1:]:
                average_sign = sum(1 if commutes(q, label) else -1 for q in group)
                self.assertEqual(average_sign, len(group) if label == target else 0)

    def test_modulation_gap_classification(self):
        self.assertEqual(classify_coefficient(dict(C_hat=.002, epsilon_C=.002), .01), 'static')
        self.assertEqual(classify_coefficient(dict(C_hat=-.008, epsilon_C=.002), .01), 'modulated')
        for estimate in (.004999, .005, .005001):
            with self.subTest(estimate=estimate):
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    classify_coefficient(dict(C_hat=estimate, epsilon_C=.002), .01)
        with self.assertRaisesRegex(ValueError, 'must be less than C_min/4'):
            classify_coefficient(dict(C_hat=0., epsilon_C=.0025), .01)

    def test_mixed_pipeline_at_minimum_amplitude(self):
        source = ThreeQubitOracle((
            PeriodicPauliTerm('XII', .72, 0.),
            PeriodicPauliTerm('IXX', .36, .04, 1.217),
            PeriodicPauliTerm('ZXY', .145, -.01, 1.463),
        ))
        with patch.object(source, 'esu_phase', wraps=source.esu_phase) as esu:
            result = run(source=source, C_min=.01, verbose=False)
        self.assertEqual(result['term_types'],
                         {'XII':'static', 'IXX':'modulated', 'ZXY':'modulated'})
        self.assertEqual({call.args[0] for call in esu.call_args_list}, {'IXX', 'ZXY'})
        self.assertTrue(result['frequency']['ZXY']['success'])
        for refined in result['refinement'].values():
            self.assertTrue(refined['diagnostics']['final_intervals_cover_truth'])

    def test_all_static_pipeline_skips_frequency_search(self):
        source = ThreeQubitOracle(tuple(PeriodicPauliTerm(t.label, t.B, 0.) for t in TERMS))
        with patch('run_all.learn_omega_for_all_targets') as search:
            result = run(source=source, C_min=.01, shots=10**9, verbose=False)
        search.assert_not_called()
        self.assertEqual(set(result['term_types'].values()), {'static'})
        for label, frequency in result['frequency'].items():
            self.assertIsNone(frequency['omega_hat'])
            self.assertEqual(frequency['esu_queries'], 0)
            refined = result['refinement'][label]
            self.assertEqual(refined['C_hat_ref'], 0.)
            self.assertEqual(refined['C']['history'], [])
            self.assertTrue(refined['B']['history'])
            self.assertTrue(refined['diagnostics']['final_intervals_cover_truth'])

    def test_matrix_order_and_noncommutation(self):
        np.testing.assert_allclose(PAULIS['XII'],np.kron(I2,np.kron(I2,X)))
        np.testing.assert_allclose(PAULIS['ZXY'],np.kron(Y,np.kron(X,Z)))
        self.assertTrue(commutes('XII','IXX'))
        self.assertFalse(commutes('XII','ZXY'))
        self.assertFalse(commutes('IXX','ZXY'))
        for p in PAULIS.values():
            np.testing.assert_allclose(p@p,I8,atol=1e-14)

    def test_all_Bell_labels_and_readout_mapping(self):
        for label,U in PAULIS.items():
            probabilities=bell_probabilities(U)
            self.assertAlmostEqual(probabilities[LABELS.index(label)],1.)
            measured=['0']*6
            for i,c in enumerate(label):
                anc,sys={'I':'00','Z':'01','X':'10','Y':'11'}[c]
                measured[i]=sys;measured[i+3]=anc
            self.assertEqual(six_bitstring_to_pauli_label(''.join(measured[::-1])),label)

    def test_explicit_six_qubit_Bell_distribution(self):
        H=ThreeQubitOracle().hamiltonian(.37)
        U=expm(-1j*.23*H)
        state=np.zeros(64,complex)
        state[9*np.arange(8)]=1/np.sqrt(8)
        state=(state.reshape(8,8)@U.T).reshape(64)
        indices=np.arange(64)
        for q in range(3):
            # CNOT(q -> q+3), then H(q), the inverse Bell transform.
            destinations=indices ^ (((indices>>q)&1)<<(q+3))
            out=np.zeros_like(state);out[destinations]=state;state=out
            zero=indices[(indices&(1<<q))==0]
            one=zero|(1<<q)
            out=state.copy()
            out[zero]=(state[zero]+state[one])/np.sqrt(2)
            out[one]=(state[zero]-state[one])/np.sqrt(2)
            state=out
        expected=bell_probabilities(U)
        for index,amplitude in enumerate(state):
            label=six_bitstring_to_pauli_label(format(index,'06b'))
            self.assertAlmostEqual(abs(amplitude)**2,expected[LABELS.index(label)],12)

    def test_reshaping_groups(self):
        for target in SUPPORT:
            group,generators=reshaping_group(target,SUPPORT)
            for label in SUPPORT:
                avg=sum(PAULIS[q]@PAULIS[label]@PAULIS[q] for q in group)/len(group)
                np.testing.assert_allclose(avg,PAULIS[label] if label==target else np.zeros((8,8)),atol=1e-12)

    def test_generalized_Ramsey_signs(self):
        source=ThreeQubitOracle()
        for label in SUPPORT:
            state,R,S,_=ramsey_probe(label)
            np.testing.assert_allclose(R@state,state,atol=1e-14)
            for theta in (-2.,-.1,0.,.7,2.5):
                d=source.ramsey_at_phase(label,theta,None,np.random.default_rng(1))
                self.assertAlmostEqual(d['R'],np.cos(2*theta),12)
                self.assertAlmostEqual(d['S'],np.sin(2*theta),12)
                U=expm(-1j*theta*PAULIS[label])
                psi=U@state
                self.assertAlmostEqual(np.vdot(psi,S@psi).real,d['S'],12)

    def test_structure_and_estimated_compensation(self):
        r=learn_structure()
        self.assertEqual(set(r['support']),set(SUPPORT))
        self.assertTrue(r['theorem_sample_budget_met'])
        # Deliberately inaccurate learned XII parameters, never exact-truth cancellation.
        comp=(PeriodicPauliTerm('XII',.7,.07,1.02),)
        r=learn_structure(known_terms=comp)
        self.assertEqual(set(r['support']),set(SUPPORT))
        r=learn_structure(windows=2,shots_per_window=200000)
        self.assertFalse(r['theorem_sample_budget_met'])
        self.assertIsNone(r['failure_probability'])

    def test_Richardson_doubled_times_and_signed_models(self):
        models=[ThreeQubitOracle(),ThreeQubitOracle((
            PeriodicPauliTerm('XII',-.61,-.073,1.07),
            PeriodicPauliTerm('IXX',.33,.038,1.23),
            PeriodicPauliTerm('ZXY',-.14,-.018,1.43)))]
        for source in models:
            results=learn_B_and_C_for_all_targets(SUPPORT,source=source)
            for label,r in results.items():
                truth=source.by_label[label]
                np.testing.assert_allclose(np.array(r['ramsey_times'][1:])/r['ramsey_times'][:-1],2.)
                self.assertLessEqual(abs(r['B_hat']-truth.B),r['epsilon_B'])
                self.assertLessEqual(abs(r['C_hat']-truth.C),r['epsilon_C'])
                self.assertLessEqual(abs(r['B_plus_C_hat']-truth.B-truth.C),r['epsilon_S'])

    def test_finite_pulse_observable_budget(self):
        source=ThreeQubitOracle()
        pulse=PulseRamseyOracle(source,SUPPORT,quadrature_bias=.01)
        for label in SUPPORT:
            d=pulse.ramsey(label,.15,None,np.random.default_rng(1))
            exact=source.ramsey(label,.15,None,np.random.default_rng(1))
            self.assertLessEqual(d['quadrature_bias_bound'],.01*(1+1e-12))
            for q in ('R','S'):
                self.assertLessEqual(abs(d[q]-exact[q]),d['quadrature_bias_bound']+1e-12)

    def test_finite_shots_and_invalid_inputs(self):
        source=ThreeQubitOracle()
        for seed in range(5):
            records=learn_B_and_C_for_all_targets(SUPPORT,shots=10**10,seed=seed)
            for label,r in records.items():
                term=source.by_label[label]
                self.assertLessEqual(abs(r['B_hat']-term.B),r['epsilon_B'])
                self.assertLessEqual(abs(r['C_hat']-term.C),r['epsilon_C'])
                choose_search_design(r,1.,1.5)
        with self.assertRaises(ValueError):phase_error(2,.01)
        with self.assertRaises(ValueError):learn_B_and_C_for_target('XII',SUPPORT,t_short=2.)
        with self.assertRaises(ValueError):learn_structure(shots_per_window=2.5)
        with self.assertRaises(ValueError):reshaping_group('III',SUPPORT)

    def test_QSP_gradient(self):
        phases=np.random.default_rng(2).normal(size=6)
        theta=np.array([0.,.2,.3,1.3,1.5])
        _,jac=_qsp_fit_residual_jacobian(phases,theta,3)
        for j in range(len(phases)):
            d=np.zeros_like(phases);d[j]=1e-6
            vp=_qsp_fit_residual_jacobian(phases+d,theta,3)[0]
            vm=_qsp_fit_residual_jacobian(phases-d,theta,3)[0]
            np.testing.assert_allclose(jac[:,j],(vp-vm)/2e-6,atol=1e-8)

    def test_complete_QSP_BBHT_and_refinement(self):
        bc=learn_B_and_C_for_all_targets(SUPPORT)
        f=learn_omega_for_all_targets(bc,verbose=False)
        self.assertAlmostEqual(sum(x['search_failure_probability_bound'] for x in f.values()),.05)
        for label,r in f.items():
            self.assertTrue(r['filter_check']['passed'])
            self.assertTrue(r['source_intervals_cover_truth'])
            self.assertTrue(r['actual_response_passes_promises'])
            self.assertTrue(r['success'])
            self.assertLessEqual(r['bbht_rounds'], r['max_bbht_rounds'])
            self.assertLessEqual(r['qsp_false_accept_probability_bound'], r['delta_filter'])
            self.assertLessEqual(r['bbht_miss_probability_bound'], r['delta_search'])
            self.assertLessEqual(r['absolute_error'],r['frequency_error_bound'])
            design = r['design']
            np.testing.assert_allclose(r['frequency_error_bound'], design.r_out / design.T_total,
                                       rtol=1e-12, atol=0)
            U=build_qsp_oracle_blocks(r['theta_values'],r['qsp_phases'])
            np.testing.assert_allclose(np.einsum('kji,kjl->kil',U.conj(),U),
                                       np.broadcast_to(np.eye(2),U.shape),atol=1e-12)
            for seed in range(20):
                found=bbht_amplitude_search(U,r['qsp_degree'],seed=seed)
                self.assertLess(abs(r['omega_bins'][found['k_hat']]-r['true_omega']),r['frequency_error_bound'])
        ref=refine_all_targets(f)
        for r in ref.values():self.assertTrue(r['diagnostics']['final_intervals_cover_truth'])
        # Externally provided tighter frequency intervals; not a claim about default search.
        source=ThreeQubitOracle()
        precise={lab:dict(record,omega_hat=source.by_label[lab].omega,frequency_error_bound=1e-6)
                 for lab,record in bc.items()}
        for r in refine_all_targets(precise).values():
            self.assertTrue(r['diagnostics']['final_intervals_cover_truth'])
            self.assertEqual(r['C']['status'],'target_reached')

    def test_mismatch_bound(self):
        for label in SUPPORT:
            c=ThreeQubitOracle().by_label[label].C
            p=normalize_input(dict(B_hat=.3,C_hat=c,epsilon_B=.01,epsilon_C=.001,
                omega_hat=1.2,frequency_error_bound=.02,omega_min=1.,omega_max=1.5))
            for m in (1,2,8):
                t=2*np.pi*m/1.2
                for w in np.linspace(1.18,1.22,7):
                    g,f=square_integrals(1.2,w,t)
                    self.assertLessEqual(abs(c/w*np.sin(w*t)),systematic_phase_bound('B',t,p)+1e-12)
                    self.assertLessEqual(abs(c*f-2*c*t/np.pi),systematic_phase_bound('C',t,p)+1e-12)

if __name__=='__main__':unittest.main()
