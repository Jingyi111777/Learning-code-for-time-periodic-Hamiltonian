"""Scientific regression tests. Run from this folder: python -m unittest -v."""
import unittest
import numpy as np
from scipy.linalg import expm
from three_qubit_model import (TERMS,PAULIS,LABELS,I8,X,Y,Z,I2,ThreeQubitOracle,
    PeriodicPauliTerm,EstimationPriors,commutes,reshaping_group,ramsey_probe,PulseRamseyOracle)
from structure_learning import learn_structure,bell_probabilities,six_bitstring_to_pauli_label
from bc_estimation import learn_B_and_C_for_all_targets,learn_B_and_C_for_target
from omega_adaptive import (learn_omega_for_all_targets,choose_search_design,
    build_qsp_oracle_blocks,bbht_amplitude_search,_qsp_fit_residual_jacobian)
from bc_refinement import refine_all_targets,Config,normalize_input,systematic_phase_bound
from numerical_utils import square_integrals,phase_error

SUPPORT=('XII','IXX','ZXY')

class Validation(unittest.TestCase):
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
            self.assertLessEqual(r['absolute_error'],r['frequency_error_bound'])
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
