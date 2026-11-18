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

class PhysicsTests(unittest.TestCase):

    def test_earth_coast_conservation(self):
        r=np.array([RE+185000.,0.,0.]);v=np.array([0.,np.sqrt(ME/np.linalg.norm(r)),0.])
        period=2*np.pi*np.sqrt(np.linalg.norm(r)**3/ME)
        a=advance(r,v,0.,period,30000.,0.,0.,313.,np.zeros(3),0.,0.)
        self.assertLess(np.linalg.norm(a[0]-r),10.)
        self.assertLess(abs(np.linalg.norm(a[1])-np.linalg.norm(v)),.01)

    def test_thrust_mass_and_integral(self):
        s=e.reset(7,False);m=e.mass(s)
        a=advance(s['r'],s['v'],0.,40.,m,STAGE_FUEL,T0,ISP0,np.array([0.,1.,0.]),s['phase'])
        used=T0*40/(G0*ISP0)
        self.assertAlmostEqual(a[3],used,places=6)
        self.assertAlmostEqual(a[4],G0*ISP0*np.log(m/(m-used)),places=7)

    def test_fuel_depletion_coasts(self):
        s=e.reset(7,False);m=e.mass(s)
        a=advance(s['r'],s['v'],0.,20.,m,10.,T0,ISP0,np.array([0.,0.,1.]),s['phase'])
        self.assertAlmostEqual(a[3],10.,places=7)
        self.assertAlmostEqual(a[2],20.,places=7)

    def test_separation_no_impulse(self):
        s=e.reset(8,False);old=e.mass(s);r=s['r'].copy();v=s['v'].copy()
        expected=advance(r,v,0.,1.,CSM_WET,0.,0.,ISP1,np.zeros(3),s['phase'])
        e.step(s,2,np.zeros(4))
        np.testing.assert_allclose(s['r'],expected[0],atol=1.e-7)
        np.testing.assert_allclose(s['v'],expected[1],atol=1.e-7)
        self.assertAlmostEqual(old-e.mass(s),STAGE_INERT+STAGE_FUEL,places=7)

    def test_surface_event(self):
        a=advance(np.array([RE+100.,0.,0.]),np.array([-1000.,0.,0.]),0.,10.,
                  10000.,0.,0.,313.,np.zeros(3),0.)
        self.assertEqual(a[5],1);self.assertLess(a[2],.15)

    def test_no_phantom_delivered_impulse(self):
        with self.assertRaises(ValueError):
            advance(np.array([RE+100000.,0.,0.]),np.zeros(3),0.,1.,1000.,1.,
                    10.,313.,np.zeros(3),0.)

    def test_lunar_orbit_and_step_refinement(self):
        j,jv=moon(0.,0.)
        r=j+np.array([RM+200000,0.,0.]);v=jv+np.array([0.,np.sqrt(MM/(RM+200000)),0.])
        a=advance(r,v,0.,86400.,CSM_DRY,0.,0.,ISP1,np.zeros(3),0.)
        b=advance(r,v,0.,86400.,CSM_DRY,0.,0.,ISP1,np.zeros(3),0.,accuracy=.5)
        self.assertEqual(a[5],0);self.assertGreater(a[-1],86399)
        self.assertLess(np.linalg.norm(a[0]-b[0]),100.)

    def test_upper_stage_one_remaining_ignition(self):
        s=e.reset(10,False)
        e.step(s,1,np.array([0.,0.,1.,-8.]))
        self.assertTrue(e.available(s)[1])
        e.step(s,0,np.array([0.,0.,0.,-10.]))
        self.assertFalse(e.available(s)[1])
        self.assertTrue(e.available(s)[2])

    def test_coordinate_frame_does_not_change_physics(self):
        a=e.reset(12,False,'inertial');b=e.reset(12,False,'local')
        axes=e.basis(b)
        np.testing.assert_allclose(axes.T@axes,np.eye(3),atol=1.e-12)
        raw=np.array([.3,.7,-.2,-3.])
        inertial=raw.copy();inertial[:3]=axes@raw[:3]
        e.step(a,1,inertial);e.step(b,1,raw)
        np.testing.assert_allclose(a['r'],b['r'],atol=1.e-7)
        np.testing.assert_allclose(a['v'],b['v'],atol=1.e-7)
