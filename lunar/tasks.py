"""independent rl tasks. initial capture states are sampled.

orbital quantities below
define the reward/acceptance criteria only; policy inputs remain navigation state.
"""
import numpy as np
from . import environment as base
from .physics import *

HANDOFF=12*RM
RESERVE=.5*SPS_FUEL

class TaskEnvironment:
    OBS=25
    def __init__(self,task,compress_dry=False,departure_energy=False,departure_approach=False):
        if task not in ('departure','capture'):raise ValueError(task)
        self.task=task
        self.compress_dry=compress_dry
        self.departure_energy=departure_energy
        self.departure_approach=departure_approach
        if departure_approach and (task!='departure' or departure_energy):
            raise ValueError('Approach reward is a separate departure ablation')
    def reset(self,seed,errors=True,frame='local'):
        s=base.reset(seed,errors,frame='local')
        s['task']=self.task
        if self.task=='departure':
            s.update(coast_interval=(10.,1800.),max_decisions=4096)
            if self.departure_approach:s['best_approach']=self.approach_score(s)
        else:
            rng=s['rng']
            radial=rng.normal(size=3);radial/=np.linalg.norm(radial)
            tangent=rng.normal(size=3);tangent-=radial*np.dot(tangent,radial)
            tangent/=np.linalg.norm(tangent)
            radius=rng.uniform(10,14)*RM
            vinf=rng.uniform(400.,1000.)
            speed=np.sqrt(2*MM/radius+vinf**2)
            tangential_fraction=rng.uniform(.2,.45)
            relative_v=speed*(-np.sqrt(1-tangential_fraction**2)*radial+tangential_fraction*tangent)
            j,jv=moon(0.,s['phase'])
            s.update(r=j+radius*radial,v=jv+relative_v,stage=1,f0=0.,
                     frame='moon_local',f1=rng.uniform(.5,1.)*SPS_FUEL,
                     burn_interval=(.1,120.),coast_interval=(1.,900.),
                     horizon=2*86400.,max_decisions=4096)
        return s
    def observation(self,s):
        if self.task=='departure':
            x=base.observation(s)
            if self.departure_approach:x[24]=s['best_approach']/30.
            return x
        j,jv=moon(s['t'],s['phase']);axes=base.basis(s).T
        return np.concatenate((axes@(s['r']-j)/(12*RM),axes@(s['v']-jv)/2000.,
            axes@s['r']/AM,axes@s['v']/10000.,axes@j/AM,axes@jv/1000.,
            [s['f1']/SPS_FUEL,base.mass(s)/CSM_WET,float(s['lit']),s['t']/s['horizon'],
             s['hold']/21600.,0.,1.])).astype(np.float32)
    def available(self,s):return base.available(s)
    def mass(self,s):return base.mass(s)
    def approach_score(self,s):
        """score only an actually visited state.

        nearby arrivals additionally need tolerable relative energy and perilune.
        smooth proximity weighting prevents remote osculating elements dominating.
        """
        j,_=moon(s['t'],s['phase']);d=np.linalg.norm(s['r']-j)
        energy,rp,_=lunar_elements(s['r'],s['v'],s['t'],s['phase'])
        near=1./(1.+(d/(12*RM))**2)
        energy_error=max(0.,-energy,energy-500000.)/500000.
        perilune_error=max(0.,RM+50000.-rp,rp-(RM+5000000.))/RM
        return float(-4*np.log1p(d/RM)-near*(2*np.log1p(energy_error)
                     +2*np.log1p(perilune_error)))
    def potential(self,s):
        j,_=moon(s['t'],s['phase']);distance=np.linalg.norm(s['r']-j)
        if self.task=='departure':
            if self.departure_approach:
                return s['best_approach']-4*(1-s['f1']/SPS_FUEL)
            p=-4*np.log1p(distance/RM)-4*(1-s['f1']/SPS_FUEL)
            if self.departure_energy:
                energy=.5*np.dot(s['v'],s['v'])-ME/np.linalg.norm(s['r'])
                # broad physical high-earth-orbit energy band for the reward.
                error=max(0.,-1600000.-energy,energy+400000.)/1.e6
                p-=4*np.log1p(error)
            return p
        energy,rp,_=lunar_elements(s['r'],s['v'],s['t'],s['phase'])
        # a safe bound orbit around a=-mu/(2e)=6131 km has this target energy.
        return (-2*np.log1p(abs(energy+400000.)/100000.)
                -2*np.log1p(max(0.,RM+50000-rp)/(RM*.1))
                -np.log1p(max(0.,distance-(RM+10000000.))/RM))
    def departure_goal(self,s):
        j,jv=moon(s['t'],s['phase']);dr=s['r']-j;dv=s['v']-jv
        energy,rp,_=lunar_elements(s['r'],s['v'],s['t'],s['phase'])
        return bool(np.linalg.norm(dr)<=HANDOFF and np.dot(dr,dv)<0 and
                    0<energy<=.5*1000.**2 and RM+50000<=rp<=RM+5000000
                    and s['f1']>=RESERVE and s['stage']==1 and not s['event'])
    def step(self,s,mode,raw,variant='distance',moon_mu=MM):
        before=self.potential(s);old_return=s['return_']
        _,_,term,trunc=base.step(s,mode,raw,variant,moon_mu)
        # with stage 1 completely dry there are no remaining controllable forces.
        # numerically propagate that physical outcome.
        if self.task=='capture' and self.compress_dry:
            while s['f1']<=1.e-6 and not (term or trunc):
                _,_,term,trunc=base.step(s,0,np.array([0.,0.,0.,30.]),variant,moon_mu)
                s['steps']-=1
                trunc=s['steps']>=s.get('max_decisions',4096) and not term
                s['auto_coast_steps']=s.get('auto_coast_steps',0)+1
        reserve_failure=False
        if self.task=='departure':
            if self.departure_approach:
                s['best_approach']=max(s['best_approach'],self.approach_score(s))
            s['success']=self.departure_goal(s)
            reserve_failure=s['f1']<RESERVE
            term=bool(s['success'] or s['event'] or s['t']>=END or reserve_failure)
            trunc=s['steps']>=s['max_decisions'] and not term
        reward=float(self.potential(s)-before+50*s['success']-25*bool(s['event'])-20*reserve_failure)
        s['return_']=old_return+reward
        return self.observation(s),reward,term,trunc
    def summary(self,s):
        row=base.summary(s)
        row['phase_success']=bool(s['success'])
        row['capture_candidate']=bool(s['success'] and self.task=='capture')
        row['task']=self.task
        row['auto_coast_steps']=s.get('auto_coast_steps',0)
        if self.departure_approach:row['best_approach_score']=s['best_approach']
        if s['success']:
            row['terminal_state']={k:(s[k].tolist() if isinstance(s[k],np.ndarray) else s[k])
                                    for k in ('r','v','t','phase','f0','f1','stage','hold')}
        return row
