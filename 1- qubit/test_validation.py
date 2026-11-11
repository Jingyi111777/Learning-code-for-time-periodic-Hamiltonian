"""Scientific regression checks; python -m unittest -v test_validation.py"""
import unittest
import numpy as np
from scipy.integrate import quad
from learning_common import (ModelOracle,Priors,PulseRamseyOracle,phase_error,
                             interval_update,square_integrals)
from bc_estimation import learn_BC
from x_coefficient_estimation import learn_x_coefficient
from structure_learning import learn_structure
import omega_adaptive as frequency
import bc_refinement as refinement

class Validation(unittest.TestCase):
    def test_ESU_integral(self):
        w,v,t=9.1,9.8,2.3
        switches=(2*np.arange(100)+1)*np.pi/(2*w)
        pts=switches[switches<t]
        g,f=square_integrals(w,v,t)
        direct=quad(lambda x: np.sign(np.cos(w*x))*np.cos(v*x),0,t,points=pts,epsabs=1e-11)[0]
        self.assertAlmostEqual(f,direct,11)
        g,f=square_integrals(w,w,2*np.pi*4/w)
        self.assertAlmostEqual(g,0.,12)
        self.assertAlmostEqual(f,16/w,12)

    def test_BC_signed_models_and_bounds(self):
        for a,b,c,w in ((.8,.3,.1,9.8),(-.4,-.25,-.1,8.6),(.2,.05,.08,10.9)):
            o=ModelOracle(a,b,c,w)
            r=learn_BC(o)
            self.assertLessEqual(abs(r['B_hat']-b),r['epsilon_B'])
            self.assertLessEqual(abs(r['C_hat']-c),r['epsilon_C'])
            self.assertLessEqual(abs(r['B_plus_C_hat']-b-c),r['epsilon_S'])
        # Known B=0 branch, no invalid t0 division.
        r=learn_BC(ModelOracle(B=0.),Priors(B_max=0.))
        self.assertEqual(r['B_hat'],0.)
        self.assertEqual(r['epsilon_B'],0.)

    def test_finite_shots_and_input_failure(self):
        for seed in range(10):
            r=learn_BC(shots=10**9,seed=seed)
            self.assertLessEqual(abs(r['B_hat']-.3),r['epsilon_B'])
            self.assertLessEqual(abs(r['C_hat']-.1),r['epsilon_C'])
        with self.assertRaises(ValueError): phase_error(1,.01)
        with self.assertRaises(ValueError): learn_BC(t_short=1.)
        with self.assertRaises(ValueError): learn_BC(T_B=.1)
        with self.assertRaises(ValueError): learn_BC(shots=3.2)

    def test_X_phase_sign_and_aliases(self):
        for A in (-.95,-.4,0.,.8,.95):
            r=learn_x_coefficient(oracle=ModelOracle(A=A),shots=100000)
            lo,hi=r['final_interval']
            self.assertLessEqual(lo,A)
            self.assertGreaterEqual(hi,A)
            self.assertLessEqual(r['error_bound'],.01)
        intervals=interval_update([(-1.,1.)],0.,10.,.01)
        self.assertGreater(len(intervals),1)

    def test_deterministic_pulse_budget(self):
        o=ModelOracle()
        p=Priors()
        pulse=PulseRamseyOracle(o,p,quadrature_bias=.004)
        for target in ('X','Z'):
            for t in (.08,.6):
                d=pulse.ramsey(target,t,None,np.random.default_rng(1))
                ideal=o.ramsey(target,t,None,np.random.default_rng(1))
                self.assertLessEqual(d['quadrature_bias_bound'],.004*(1+1e-12))
                for key in ('R','S'):
                    self.assertLessEqual(abs(d[key]-ideal[key]),d['quadrature_bias_bound']+1e-12)
        r=learn_BC(pulse,p,T_B=2.,quadrature_bias=.004)
        self.assertLessEqual(abs(r['C_hat']-.1),r['epsilon_C'])

    def test_structure_random_windows_and_compensation(self):
        r=learn_structure()
        self.assertEqual(r['support'],['X','Z'])
        self.assertTrue(r['theorem_sample_budget_met'])
        # Z coefficient vanishes exactly at the origin; random windows recover it.
        r=learn_structure(ModelOracle(B=.1,C=-.1))
        self.assertEqual(r['support'],['X','Z'])
        from learning_common import X
        r=learn_structure(known_support=('X',),h_hat=.8*X)
        self.assertEqual(r['support'],['X','Z'])
        r=learn_structure(windows=1,shots_per_window=5000)
        self.assertFalse(r['theorem_sample_budget_met'])
        self.assertIsNone(r['failure_probability'])

    def test_QSP_gradient(self):
        phases=np.random.default_rng(1).normal(size=6)
        theta=np.array([0.,.1,.3,1.4,1.5])
        value,jac=frequency._qsp_fit_residual_jacobian(phases,theta,3)
        for j in range(len(phases)):
            d=np.zeros(len(phases));d[j]=1e-6
            vp=frequency._qsp_fit_residual_jacobian(phases+d,theta,3)[0]
            vm=frequency._qsp_fit_residual_jacobian(phases-d,theta,3)[0]
            np.testing.assert_allclose(jac[:,j],(vp-vm)/2e-6,atol=1e-8)

    def test_QSP_BBHT_refinement(self):
        bc=learn_BC()
        f=frequency.learn_omega_for_single_qubit(bc,verbose=False)
        self.assertTrue(f['filter_check']['passed'])
        self.assertTrue(f['source_intervals_cover_truth'])
        self.assertTrue(f['actual_response_passes_promises'])
        self.assertLessEqual(abs(f['omega_hat']-9.8),f['frequency_error_bound'])
        U=frequency.build_qsp_oracle_blocks(f['theta_values'],f['qsp_phases'])
        np.testing.assert_allclose(np.einsum('kji,kjl->kil',U.conj(),U),
                                   np.broadcast_to(np.eye(2),(len(U),2,2)),atol=1e-12)
        for seed in range(50):
            found=frequency.bbht_amplitude_search(U,f['qsp_degree'],seed=seed)
            # This is a regression observation, not a proof of success probability.
            self.assertLess(abs(f['omega_bins'][found['k_hat']]-9.8),f['frequency_error_bound'])
        r=refinement.refine_single_qubit(f)
        self.assertTrue(r['diagnostics']['final_intervals_cover_truth'])
        self.assertNotEqual(r['C']['status'],'target_reached')
        # Propagate a more precise externally supplied frequency interval.
        precise=dict(bc,omega_hat=9.8,frequency_error_bound=1e-5)
        r=refinement.refine_single_qubit(precise)
        self.assertTrue(r['diagnostics']['final_intervals_cover_truth'])
        self.assertEqual(r['C']['status'],'target_reached')

    def test_frequency_mismatch_bound(self):
        for ew in (.0001,.01,.1):
            p=refinement.normalize_input(dict(B_hat=.3,C_hat=.1,epsilon_B=.01,
                epsilon_C=.01,omega_hat=9.8,frequency_error_bound=ew,
                omega_min=8.5,omega_max=11.))
            for m in (1,2,8,32):
                t=2*np.pi*m/9.8
                for w in np.linspace(9.8-ew,9.8+ew,9):
                    for C in (-.11,.11):
                        actual_B=abs(C/w*np.sin(w*t))
                        g,f=square_integrals(9.8,w,t)
                        actual_C=abs(C*f-2*C*t/np.pi)
                        self.assertLessEqual(actual_B,refinement.systematic_phase_bound('B',t,p)+1e-12)
                        self.assertLessEqual(actual_C,refinement.systematic_phase_bound('C',t,p)+1e-12)

if __name__=='__main__':
    unittest.main()
