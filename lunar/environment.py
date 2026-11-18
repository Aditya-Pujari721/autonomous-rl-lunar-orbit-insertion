"""exact-navigation pilot."""
import numpy as np
from .physics import *

OBS = 25

def reset(seed, errors=True, frame='inertial'):
    rng=np.random.default_rng(int(seed))
    angle=rng.uniform(0,2*np.pi)
    r=(RE+185000)*np.array([np.cos(angle),np.sin(angle),0.])
    v=np.sqrt(ME/np.linalg.norm(r))*np.array([-np.sin(angle),np.cos(angle),0.])
    return dict(r=r,v=v,t=0.,phase=rng.uniform(0,2*np.pi),stage=0,frame=frame,seed=int(seed),
                f0=STAGE_FUEL,f1=SPS_FUEL,lit=False,spent=False,rng=rng,
                gain=1+rng.normal(0,.002) if errors else 1.,errors=errors,
                closest=1.e30,max_radius=np.linalg.norm(r),burns=0,dv=0.,
                event=0,steps=0,hold=0.,success=False,return_=0.)

def mass(s):
    return CSM_DRY+s['f1']+(STAGE_INERT+s['f0'] if s['stage']==0 else 0)

def basis(s):
    if s.get('frame','inertial')=='inertial':return np.eye(3)
    r,v=s['r'],s['v']
    if s.get('frame')=='moon_local':
        j,jv=moon(s['t'],s['phase'],s.get('orbit_radius',AM));r=r-j;v=v-jv
    radial=r/np.linalg.norm(r)
    normal=np.cross(r,v)
    if np.linalg.norm(normal)<1.e-9:
        axis=np.eye(3)[np.argmin(np.abs(radial))]
        normal=np.cross(radial,axis)
    normal/=np.linalg.norm(normal)
    tangent=np.cross(normal,radial)
    return np.stack((radial,tangent,normal),axis=1)

def observation(s):
    j,jv=moon(s['t'],s['phase'],s.get('orbit_radius',AM))
    axes=basis(s).T
    r,v,j,jv=axes@s['r'],axes@s['v'],axes@j,axes@jv
    # global, fixed unit scales only. the two position scales retain leo resolution.
    return np.concatenate((r/AM,v/10000,j/AM,jv/1000,
        np.arcsinh(r/RE),np.arcsinh((r-j)/RM),
        [(s['f0']/STAGE_FUEL),s['f1']/SPS_FUEL,s['stage'],float(s['lit']),
         float(s['spent']),s['t']/END,0.])).astype(np.float32)

def available(s):
    burn=(s['f0']>1.e-6 and not s['spent']) if s['stage']==0 else s['f1']>1.e-6
    return np.array([True,burn,s['stage']==0])

def potential(s, variant='distance'):
    j,_=moon(s['t'],s['phase'],s.get('orbit_radius',AM))
    d=np.linalg.norm(s['r']-j)
    # physical goal only.
    p=-4*np.log1p(d/RM)
    if variant=='capture':
        energy,rp,ra=lunar_elements(s['r'],s['v'],s['t'],s['phase'],s.get('orbit_radius',AM))
        proximity=np.exp(-d/(20*RM))
        p-=proximity*(np.log1p(max(0.,energy)/(MM/RM))+
                       np.log1p(max(0.,RM+50000-rp)/RM))
    return float(p)

def step(s, mode, raw, variant='distance', moon_mu=MM):
    """mode 0 coast, 1 burn, 2 separate. raw: inertial xyz + duration latent.

    consecutive j2 burn decisions continue one ignition. a coast closes it;
    its single remaining mission ignition cannot then be reused. sps restarts.
    invalid actions are excluded by the policy's hardware mask.
    """
    if not available(s)[mode]:
        raise ValueError('Unavailable hardware action')
    before=potential(s,variant)
    frac=1/(1+np.exp(-np.clip(raw[3],-30,30)))
    lo,hi=s.get('burn_interval',(1.,400.)) if mode==1 else s.get('coast_interval',(30.,21600.))
    duration=lo+(hi-lo)*frac
    direction=np.asarray(raw[:3],dtype=float)
    direction=direction/max(1.e-12,np.linalg.norm(direction))
    direction=basis(s)@direction
    thrust=0.
    isp=ISP0 if s['stage']==0 else ISP1
    fuel=s['f0'] if s['stage']==0 else s['f1']
    if mode==2:
        s['stage']=1; s['f0']=0.; s['lit']=False
        duration=1.;fuel=s['f1'];isp=ISP1
    elif mode==1:
        if not s['lit']: s['burns']+=1
        s['lit']=True
        thrust=(T0 if s['stage']==0 else T1)*s['gain']
        if s['errors']:
            direction+=s['rng'].normal(0,.001,3)
            direction/=np.linalg.norm(direction)
    elif s['lit']:
        if s['stage']==0:s['spent']=True
        s['lit']=False
    horizon=s.get('horizon',END)
    duration=min(duration,horizon-s['t'])
    result=advance(s['r'],s['v'],s['t'],duration,mass(s),fuel,thrust,isp,
                   direction,s['phase'],moon_mu,accuracy=s.get('accuracy',1.),hold=s['hold'],
                   min_orbit_alt=s.get('min_orbit_alt',50000.),max_orbit_alt=s.get('max_orbit_alt',10000000.),
                   orbit_radius=s.get('orbit_radius',AM),diagnostics=s.get('approach_metrics',None))
    r,v,t,used,dv,event,closest,maxr,hold=result
    key='f0' if s['stage']==0 else 'f1'
    s[key]=max(0.,s[key]-used)
    s['r'],s['v'],s['t']=r,v,t
    s['closest']=min(s['closest'],closest)
    s['max_radius']=max(s['max_radius'],maxr)
    s['dv']+=dv;s['event']=event;s['steps']+=1
    # candidate requires six hours of continuously sampled bounded orbit, followed
    # by a separate 24h unpowered verification in evaluation.
    s['hold']=hold
    s['success']=s['hold']>=21600 and not event
    if 'record_step' in s:
        s['record_step'](s, direction, used, flow_rate=thrust/(G0*isp))
    terminated=bool(event or s['success'] or t>=horizon)
    truncated=s['steps']>=s.get('max_decisions',2048) and not terminated
    reward=potential(s,variant)-before + (50. if s['success'] else 0.)-(25. if event else 0.)
    s['return_']+=reward
    return observation(s),reward,terminated,truncated

def summary(s):
    result=dict(seed=s['seed'],closest_km=s['closest']/1000,max_earth_radius_km=s['max_radius']/1000,
                escaped_earth=s['max_radius']>1.e8,within_66000km=s['closest']<66.e6,
                capture_candidate=s['success'],event=s['event'],burns=s['burns'],
                decisions=s['steps'],days=s['t']/86400,delivered_dv=s['dv'],
                fuel0=s['f0'],fuel1=s['f1'],reward=s['return_'])
    return {k:v.item() if isinstance(v,np.generic) else v for k,v in result.items()}
