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

class PolicyAndIntegrationTests(unittest.TestCase):

    def test_actor_control_and_gate_receive_gradients(self):
        torch.manual_seed(42)
        actor=Actor()
        x=torch.randn(12,25);mask=torch.ones(12,3,dtype=torch.bool)
        cat,normal,value=actor(x,mask)
        mode=torch.ones(12,dtype=torch.int64);raw=normal.sample()
        loss=-(likelihood(cat,normal,mode,raw)*torch.linspace(-1,1,12)).mean()+value.square().mean()
        loss.backward()
        self.assertGreater(float(actor.control.weight.grad.norm()),0.)
        self.assertGreater(float(actor.mode.weight.grad.norm()),0.)

    def test_approach_diagnostics_do_not_change_dynamics(self):
        radius=360735539.9728;j,jv=moon(0.,0.,radius)
        rr=RM+200000.;r=j+np.array([rr,0.,0.]);v=jv+np.array([0.,np.sqrt(MM/rr),0.])
        args=(r,v,0.,300.,CSM_DRY,0.,0.,ISP1,np.zeros(3),0.)
        plain=advance(*args,orbit_radius=radius)
        metric=np.array([0.,np.inf,0.,0.,0.,0.])
        logged=advance(*args,orbit_radius=radius,diagnostics=metric)
        for a,b in zip(plain,logged):np.testing.assert_array_equal(a,b)
        self.assertAlmostEqual(metric[1],logged[6])
        self.assertAlmostEqual(metric[3],.5*metric[2]**2-MM/metric[1])
        self.assertEqual(metric[4],0.)
        self.assertEqual(metric[5],CSM_DRY)

    def test_mode_conditional_likelihood(self):
        p=Actor();x=torch.zeros(3,25);mask=torch.ones(3,3,dtype=torch.bool)
        c,n,_=p(x,mask);m=torch.tensor([0,1,2]);raw=n.sample();changed=raw.clone()
        changed[0,:4]+=99;changed[1,4]+=99;changed[2,:]+=99
        torch.testing.assert_close(likelihood(c,n,m,raw),likelihood(c,n,m,changed))
        torch.testing.assert_close((likelihood(c,n,m,raw)-likelihood(c,n,m,raw)).exp(),torch.ones(3))

    def test_episode_targets_and_time_trace(self):
        a,r=targets([0.,2.],[0.,0.],[3600.,3600.],3600.)
        np.testing.assert_allclose(r,[2.,2.]);np.testing.assert_allclose(a,[2/np.e,2.])
        a2,_=targets([0.,0.,2.],[0.,0.,0.],[1800.,1800.,3600.],3600.)
        self.assertAlmostEqual(a[0],a2[0])

    def test_circular_moon_radius_and_speed(self):
        radius=360735539.9728;r,v=moon(0.,0.,radius)
        self.assertAlmostEqual(np.linalg.norm(r),radius,places=6)
        self.assertAlmostEqual(np.dot(r,v),0.,places=6)
        self.assertAlmostEqual(np.linalg.norm(v)**2,(ME+MM)/radius,places=6)

    def test_target_orbit_hold_bounds(self):
        radius=360735539.9728;j,jv=moon(0.,0.,radius)
        rr=RM+200000.;r=j+np.array([rr,0.,0.]);v=jv+np.array([0.,np.sqrt(MM/rr),0.])
        a=advance(r,v,0.,600.,CSM_DRY,0.,0.,ISP1,np.zeros(3),0.,
                  min_orbit_alt=80000.,max_orbit_alt=350000.,orbit_radius=radius)
        self.assertEqual(a[5],0);self.assertGreater(a[-1],599.)
        b=advance(r,v,0.,600.,CSM_DRY,0.,0.,ISP1,np.zeros(3),0.,
                  min_orbit_alt=80000.,max_orbit_alt=150000.,orbit_radius=radius)
        self.assertEqual(b[-1],0.)
