"""continue capture ppo in a new run directory."""
import argparse, hashlib, json, shutil, time
from pathlib import Path
import numpy as np
import torch
from .capture_nominal import Capture, collect, load_parent, metrics
from .policy import update

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--name',default='capture_continuation')
    ap.add_argument('--parent',default='model/policy.pt')
    ap.add_argument('--updates',type=int,default=64)
    ap.add_argument('--episodes',type=int,default=32)
    ap.add_argument('--collision-cost',type=float,default=25.)
    ap.add_argument('--namespace',type=int,default=20261117)
    ap.add_argument('--risk-selection',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(2)
    if args.updates<0 or args.episodes<1:raise ValueError("Invalid training budget")
    root=Path('runs')/args.name
    root.mkdir(parents=True,exist_ok=False)
    (root/'source').mkdir()
    for p in Path('lunar').glob('*.py'):shutil.copy2(p,root/'source'/p.name)
    reserved=set(json.loads(Path('docs/provenance/reserved_seeds.json').read_text()))
    for previous in Path('runs').glob('*/split.json'):
        for values in json.loads(previous.read_text()).values():reserved.update(values)
    rng=np.random.default_rng(np.random.SeedSequence([42,args.namespace]))
    def draw(n):
        values=[]
        while len(values)<n:
            seed=int(rng.integers(0,2**32-1))
            if seed not in reserved:values.append(seed);reserved.add(seed)
        return values
    split={'development':draw(64),'test':draw(128)}
    (root/'split.json').write_text(json.dumps(split,indent=2))
    config=vars(args)|dict(seed=42,exact_navigation=True,exact_execution=True,
        min_alt_km=50,max_alt_km=10000,hold_hours=6,verification_hours=24,
        reference_trajectory=False,parent_sha256=hashlib.sha256(Path(args.parent).read_bytes()).hexdigest(),
        selection=('development captures minus twice impacts, then captures' if args.risk_selection else
                   'development captures, then fewer impacts'),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source').glob('*.py')})
    (root/'config.json').write_text(json.dumps(config,indent=2))
    torch.manual_seed(42)
    actor=load_parent(args.parent)
    opt=torch.optim.Adam(actor.parameters(),lr=1e-4,eps=1e-5)
    env=Capture(collision_cost=args.collision_cost)
    excluded=reserved
    completed=[]; best=None; started=time.time()
    for k in range(args.updates+1):
        if k:
            seeds=[]
            while len(seeds)<args.episodes:
                seed=int(rng.integers(0,2**32-1))
                if seed not in excluded:seeds.append(seed)
            buffers,rows=collect(actor,env,seeds,42+k*1000)
            completed+=rows
            kl=update(actor,opt,buffers,6*3600.)
            stats=metrics(rows)|dict(update=k,kl=kl,seconds=time.time()-started)
            with (root/'progress.jsonl').open('a') as f:f.write(json.dumps(stats)+'\n')
            print(json.dumps(stats),flush=True)
        if k%16==0 or k==args.updates:
            evaluations={}
            for name,det in [('stochastic',False),('deterministic',True)]:
                _,rows=collect(actor,env,split['development'],424242,det)
                evaluations[name]=rows
                m=metrics(rows)
                key=(m['captures'],-m['impacts'])
                if args.risk_selection:
                    key=(m['captures']-2*m['impacts'],m['captures'])
                if best is None or key>best:
                    best=key
                    torch.save(dict(model=actor.state_dict(),config=config,update=k,deterministic=det),root/'selected.pt')
                print('development',k,name,json.dumps(m),flush=True)
            (root/f'development_{k}.json').write_text(json.dumps(evaluations))
            torch.save(dict(model=actor.state_dict(),optimizer=opt.state_dict(),config=config,update=k),root/'latest.pt')
            (root/'training_episodes.json').write_text(json.dumps(completed))
    # selection is frozen before opening these new test cases.
    cp=torch.load(root/'selected.pt',weights_only=False)
    actor.load_state_dict(cp['model'])
    result={}
    for name,coast in [('own',False),('coast',True)]:
        _,rows=collect(actor,env,split['test'],8675309,cp['deterministic'],coast)
        result[name]=rows
        print('test',name,json.dumps(metrics(rows)),flush=True)
    (root/'test.json').write_text(json.dumps(result))
    (root/'complete.json').write_text(json.dumps(dict(selected_update=cp['update'],deterministic=cp['deterministic'],seconds=time.time()-started)))


if __name__=='__main__':main()
