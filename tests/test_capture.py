import unittest, copy, json
import numpy as np
import torch
from unittest.mock import patch
from lunar.physics import *
from lunar import environment as e
from lunar import environment as base
from lunar.capture_nominal import Capture, load_parent
from lunar.tasks import TaskEnvironment, RESERVE
from lunar.policy import Actor, likelihood, targets

class CaptureNominalTests(unittest.TestCase):

    def test_exact_execution_and_propellant_accounting(self):
        a=Capture();b=Capture()
        s=a.reset(42);t=copy.deepcopy(s)
        self.assertFalse(s['errors']);self.assertEqual(s['gain'],1.)
        raw=np.array([1.,0.,0.,-2.,0.])
        r,_=a.step(s,1,raw);q,_=b.step(t,1,raw)
        np.testing.assert_array_equal(s['r'],t['r'])
        self.assertEqual(r,q)
        self.assertLess(s['f1'],s['initial_fuel'])

    def test_failed_verification_does_not_terminate_or_move_state(self):
        env=Capture();s=env.reset(42);r=s['r'].copy()
        def candidate(state,*args,**kwargs):
            state['success']=True;state['hold']=21600.
            return None,50.,True,False
        with patch.object(TaskEnvironment,'step',side_effect=candidate), patch('lunar.capture_nominal.advance',return_value=(None,None,None,None,None,0,None,None,0)):
            reward,done=env.step(s,0,np.zeros(5))
        self.assertFalse(done);self.assertFalse(s['verified_orbit'])
        self.assertEqual(s['hold'],0.);self.assertEqual(reward,0.)
        np.testing.assert_array_equal(s['r'],r)

    def test_collision_cost_changes_reward_only(self):
        def collision(state,*args,**kwargs):
            state['event']=2;state['success']=False
            return None,0.,True,False
        a=Capture(collision_cost=25.);b=Capture(collision_cost=100.)
        s=a.reset(42);t=copy.deepcopy(s)
        with patch.object(TaskEnvironment,'step',side_effect=collision):
            r,done=a.step(s,0,np.zeros(5));q,end=b.step(t,0,np.zeros(5))
        self.assertTrue(done and end);self.assertAlmostEqual(r-q,75.)
        np.testing.assert_array_equal(s['r'],t['r'])



class TaskTests(unittest.TestCase):

    def test_capture_initial_states_are_incoming_and_unbound(self):
        task=TaskEnvironment('capture')
        for seed in range(100):
            s=task.reset(seed);j,jv=moon(0.,s['phase'])
            energy,_,_=lunar_elements(s['r'],s['v'],0.,s['phase'])
            self.assertGreater(energy,0.)
            self.assertLess(np.dot(s['r']-j,s['v']-jv),0.)
            self.assertGreaterEqual(s['f1'],RESERVE)
            self.assertEqual(task.observation(s).shape,(25,))
            self.assertTrue(np.isfinite(task.observation(s)).all())
            self.assertFalse(task.available(s)[2])

    def test_capture_goal_not_awarded_for_proximity(self):
        task=TaskEnvironment('capture');s=task.reset(42,False)
        task.step(s,0,np.array([0.,0.,0.,-10.]))
        self.assertFalse(s['success']);self.assertEqual(s['hold'],0.)

    def test_capture_coordinate_basis_is_orthonormal(self):
        s=TaskEnvironment('capture').reset(42)
        b=base.basis(s)
        np.testing.assert_allclose(b.T@b,np.eye(3),atol=1.e-12)

    def test_capture_horizon_and_summary(self):
        task=TaskEnvironment('capture');s=task.reset(42,False)
        s['t']=s['horizon']-.1
        _,_,term,trunc=task.step(s,0,np.zeros(4))
        self.assertTrue(term);self.assertFalse(trunc)
        self.assertAlmostEqual(s['t'],s['horizon'])
        self.assertFalse(task.summary(s)['phase_success'])

    def test_dry_coast_compression_preserves_outcome_and_return(self):
        normal=TaskEnvironment('capture');fast=TaskEnvironment('capture',True)
        a=normal.reset(42,False);a['f1']=0.;b=copy.deepcopy(a)
        raw=np.array([0.,0.,0.,30.])
        while True:
            _,_,term,trunc=normal.step(a,0,raw)
            if term or trunc:break
        fast.step(b,0,raw)
        np.testing.assert_allclose(a['r'],b['r'],atol=1.e-6)
        np.testing.assert_allclose(a['v'],b['v'],atol=1.e-8)
        self.assertAlmostEqual(a['return_'],b['return_'],places=9)
        self.assertEqual(a['success'],b['success']);self.assertEqual(a['event'],b['event'])
        self.assertEqual(b['steps'],1)
