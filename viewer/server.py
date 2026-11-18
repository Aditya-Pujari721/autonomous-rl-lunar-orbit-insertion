"""local lunar capture replay service. run with python -m viewer.server."""
import argparse
import json
import math
import secrets
from functools import lru_cache
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
import numpy as np
import torch
from lunar.capture_nominal import Capture, load_parent
from lunar.physics import AM, RM, MM, INC, OMEGA, SPS_FUEL, moon

ROOT=Path(__file__).resolve().parents[1]
STATIC=ROOT/"viewer/static"
model_dir="model"
torch.set_num_threads(2)
actor=load_parent(ROOT/model_dir/"policy.pt").eval()
lock=Lock()
bounds={"radius":[10,14],"speed":[400,1000],"side":[.2,.45],"fuel":[50,100],
        "latitude":[-90,90],"longitude":[-180,180],"heading":[-180,180],"phase":[-180,180]}

def initial(env, seed, custom):
    s=env.reset(seed)
    if custom is None:
        return s
    if not isinstance(custom,dict) or set(custom)!=set(bounds):
        raise ValueError("Enter all approach parameters.")
    p={}
    for key,(lo,hi) in bounds.items():
        value=float(custom[key])
        if not math.isfinite(value):raise ValueError("Enter finite numbers.")
        p[key]=min(hi,max(lo,value))
    lat,lon,head=np.deg2rad([p["latitude"],p["longitude"],p["heading"]])
    radial=np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
    east=np.array([-np.sin(lon),np.cos(lon),0.])
    tangent=np.cos(head)*np.cross(radial,east)+np.sin(head)*east
    s["phase"]=math.radians(p["phase"])
    j,jv=moon(0.,s["phase"])
    radius=p["radius"]*RM
    s.update(r=j+radius*radial,v=jv+math.sqrt(p["speed"]**2+2*MM/radius)*(
        -math.sqrt(1-p["side"]**2)*radial+p["side"]*tangent),
        f1=SPS_FUEL*p["fuel"]/100,initial_fuel=SPS_FUEL*p["fuel"]/100)
    s["max_radius"]=np.linalg.norm(s["r"])
    s["parameters"]=p
    return s

def track(seed, custom, coast):
    env=Capture()
    s=initial(env,seed,custom)
    def point():
        j,jv=moon(s["t"],s["phase"])
        return [s["t"],*((s["r"]-j)/RM),*((s["v"]-jv)/RM),100*s["f1"]/SPS_FUEL]
    points=[point()]
    burns=[]
    previous_dv=0.
    def record(state,direction,used,flow_rate):
        nonlocal previous_dv
        if used>0:
            burns.append([points[-1][0],used/flow_rate,*direction,s["dv"]-previous_dv,s["burns"]])
        previous_dv=s["dv"]
        points.append(point())
    s["record_step"]=record
    with torch.inference_mode():
        while True:
            mode,raw=0,np.zeros(5)
            if not coast:
                cat,normal,_=actor(torch.tensor(env.observation(s))[None],torch.tensor(env.available(s))[None])
                mode=int(cat.logits.argmax());raw=normal.mean[0].numpy()
            else: raw[4]=30.
            _,done=env.step(s,mode,raw)
            if done:break
    decision_end=s["t"]
    if s["verified_orbit"]:
        from lunar.physics import advance, ISP1
        for _ in range(144):
            r,v,t,_,_,event,*_=advance(s["r"],s["v"],s["t"],600.,env.mass(s),0.,0.,
                                       ISP1,np.zeros(3),s["phase"])
            s.update(r=r,v=v,t=t)
            points.append(point())
            if event:break
    return dict(decision_end=decision_end,points=points,burns=burns,capture=bool(s["verified_orbit"]),event=int(s["event"]),
                phase=s["phase"],parameters=s.get("parameters"))

@lru_cache(maxsize=32)
def replay(seed,params):
    custom=json.loads(params)
    return dict(seed=seed,policy=model_dir,omega=OMEGA,inclination=INC,
                flight=track(seed,custom,False),coast=track(seed,custom,True))

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args,**kw):super().__init__(*args,directory=str(STATIC),**kw)
    def do_POST(self):
        if self.path!="/api/flights":return self.send_error(404)
        try:
            size=int(self.headers.get("Content-Length",0))
            if not 0<size<=4096:raise ValueError("Invalid request size.")
            data=json.loads(self.rfile.read(size))
            if not isinstance(data,dict):raise ValueError("Expected a flight request.")
            seed=data.get("seed",secrets.randbits(32))
            if type(seed)!=int or not 0<=seed<2**32:raise ValueError("Invalid seed.")
            with lock:payload=replay(seed,json.dumps(data.get("parameters"),sort_keys=True))
            status=200
        except (ValueError,TypeError) as exc:status,payload=400,{"error":str(exc)}
        body=json.dumps(payload,allow_nan=False).encode()
        self.send_response(status);self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(body)));self.end_headers();self.wfile.write(body)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--host",default="127.0.0.1");ap.add_argument("--port",type=int,default=8765)
    args=ap.parse_args()
    ThreadingHTTPServer((args.host,args.port),Handler).serve_forever()

