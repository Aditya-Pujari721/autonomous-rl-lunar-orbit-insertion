"""si units; earth-centered inertial, circular inclined moon; finite rk4 thrust.

historical hardware approximations and omissions are documented in the project readme.
"""
import numpy as np
from numba import njit

G0 = 9.80665
ME, MM = 3.986004418e14, 4.9048695e12
RE, RM, AM = 6378137., 1737400., 384400000.
OMEGA = np.sqrt((ME + MM) / AM**3)
INC = np.deg2rad(5.145)
LB = .45359237
CSM_WET = (12392. + 51258.) * LB
SPS_FUEL = (15730.9 + 25105.1) * LB
CSM_DRY = CSM_WET - SPS_FUEL
STAGE_FUEL = (283213. - 122380.) * LB
STAGE_INERT = 122380. * LB - CSM_WET
T0, T1 = 200000. * 4.4482216152605, 20396. * 4.4482216152605
ISP0, ISP1 = 418., 313.
END = 10 * 86400.

@njit(cache=True)
def moon(t, phase, orbit_radius=AM):
    omega=np.sqrt((ME+MM)/orbit_radius**3)
    a = phase + omega*t
    r = orbit_radius*np.array([np.cos(a), np.sin(a)*np.cos(INC), np.sin(a)*np.sin(INC)])
    v = orbit_radius*omega*np.array([-np.sin(a), np.cos(a)*np.cos(INC), np.cos(a)*np.sin(INC)])
    return r, v

@njit(cache=True)
def gravity(r, t, phase, moon_mu=MM, orbit_radius=AM):
    j, _ = moon(t, phase,orbit_radius)
    d = j-r
    return -ME*r/np.linalg.norm(r)**3 + moon_mu*(d/np.linalg.norm(d)**3-j/orbit_radius**3)

@njit(cache=True)
def advance(r, v, t, duration, mass, fuel, thrust, isp, direction, phase,
            moon_mu=MM, accuracy=1., hold=0., min_orbit_alt=50000., max_orbit_alt=10000000.,orbit_radius=AM,
            diagnostics=None):
    """event-aware stepping; delivered dv excludes gravity. return consumed fuel.

    surface crossing is bracketed to <=0.05 seconds by retrying a shortened step.
    integrator step selection depends on current distances.
    """
    if thrust>0 and abs(np.linalg.norm(direction)-1.)>1.e-9:
        raise ValueError('Nonzero thrust requires a unit direction')
    end = t + duration
    used = 0.
    dv = 0.
    event = 0
    closest = np.linalg.norm(r-moon(t, phase,orbit_radius)[0])
    max_radius = np.linalg.norm(r)
    if diagnostics is not None and closest < diagnostics[1]:
        _,jv=moon(t,phase,orbit_radius)
        speed=np.linalg.norm(v-jv)
        diagnostics[:]=np.array([t,closest,speed,.5*speed**2-MM/closest,fuel,mass])
    while t < end-1.e-8:
        j, _ = moon(t, phase,orbit_radius)
        de, dm = np.linalg.norm(r), np.linalg.norm(r-j)
        if de <= RE or dm <= RM:
            event = 1 if de <= RE else 2
            break
        force = thrust if used < fuel-1.e-9 else 0.
        flow = force/(G0*isp)
        step = min(end-t, 120.*accuracy,
                   .025*accuracy*np.sqrt(de**3/ME),
                   .025*accuracy*np.sqrt(dm**3/MM))
        if force > 0:
            step = min(step, 2.*accuracy, (fuel-used)/flow)
        while True:
            m = mass-used
            a1 = gravity(r,t,phase,moon_mu,orbit_radius)+force/m*direction
            r2 = r+v*step/2
            v2 = v+a1*step/2
            a2 = gravity(r2,t+step/2,phase,moon_mu,orbit_radius)+force/(m-flow*step/2)*direction
            r3 = r+v2*step/2
            v3 = v+a2*step/2
            a3 = gravity(r3,t+step/2,phase,moon_mu,orbit_radius)+force/(m-flow*step/2)*direction
            r4 = r+v3*step
            v4 = v+a3*step
            a4 = gravity(r4,t+step,phase,moon_mu,orbit_radius)+force/(m-flow*step)*direction
            rn = r+step*(v+2*v2+2*v3+v4)/6
            vn = v+step*(a1+2*a2+2*a3+a4)/6
            jn, _ = moon(t+step,phase,orbit_radius)
            crossing = np.linalg.norm(rn)<=RE or np.linalg.norm(rn-jn)<=RM
            if crossing and step>.05:
                step /= 2
            else:
                break
        if force > 0:
            dv += G0*isp*np.log(m/(m-flow*step))
        used += flow*step
        r, v, t = rn, vn, t+step
        closest = min(closest, np.linalg.norm(r-jn))
        if diagnostics is not None and np.linalg.norm(r-jn) < diagnostics[1]:
            _,jv=moon(t,phase,orbit_radius)
            speed=np.linalg.norm(v-jv)
            diagnostics[:]=np.array([t,np.linalg.norm(r-jn),speed,
                                    .5*speed**2-MM/np.linalg.norm(r-jn),fuel-used,mass-used])
        max_radius = max(max_radius,np.linalg.norm(r))
        energy,rp,ra=lunar_elements(r,v,t,phase,orbit_radius)
        hold = hold+step if energy<0 and rp>=RM+min_orbit_alt and ra<=RM+max_orbit_alt else 0.
        if crossing:
            event = 1 if np.linalg.norm(r)<=RE else 2
            break
    return r,v,t,used,dv,event,closest,max_radius,hold

@njit(cache=True)
def lunar_elements(r,v,t,phase,orbit_radius=AM):
    j,jv = moon(t,phase,orbit_radius)
    dr,dv = r-j,v-jv
    radius = np.linalg.norm(dr)
    energy = .5*np.dot(dv,dv)-MM/radius
    h = np.cross(dr,dv)
    ecc = np.sqrt(max(0.,1+2*energy*np.dot(h,h)/MM**2))
    rp = np.dot(h,h)/(MM*(1+ecc))
    ra = -MM/(2*energy)*(1+ecc) if energy<0 else 1.e30
    return energy,rp,ra
