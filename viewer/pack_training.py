"""export recorded decision positions in lunar radii to a compact browser asset."""
import gzip
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from lunar.physics import AM, RM, INC, OMEGA

root = Path("runs/recorded_capture")
out = Path("viewer/static/training-v2.bin.gz")
stage1 = json.loads((root/"s1/runs/recorded/training_episodes.json").read_text())
order = [r["seed"] for r in stage1]
expected = set(order)

def export(source, dest, stage):
    paths = defaultdict(list)
    success = {}
    with gzip.open(source, "rt") as tape:
        for line in tape:
            row = json.loads(line)
            seed = row[0]
            if stage == 1 and seed not in expected:
                continue
            success[seed] = bool(row[-1])
            points = paths[seed]
            if not points:
                points.append([row[1],row[2],*row[3:6]])
            points.append([row[12],row[2],*row[13:16]])
    for seed in (order if stage == 1 else paths):
        points = np.asarray(paths[seed])
        if len(points)>48:
            points=points[np.linspace(0,len(points)-1,48).round().astype(int)]
        a=points[:,1]+OMEGA*points[:,0]
        x,y,z=points[:,2:].T
        y0=y*np.cos(INC)+z*np.sin(INC)
        coords=np.column_stack((x*np.cos(a)+y0*np.sin(a)-AM,-x*np.sin(a)+y0*np.cos(a),z*np.cos(INC)-y*np.sin(INC)))/RM
        dest.write(bytes([stage,len(coords),success[seed]]))
        dest.write(np.column_stack((points[:,0],coords)).astype('<f4').tobytes())
    return len(order if stage==1 else paths)

with gzip.open(out,"wb",compresslevel=3) as dest:
    counts=[export(root/"s1/commands.jsonl.gz",dest,1)]
    for stage in (2,3):
        counts.append(sum(export(p,dest,stage) for p in sorted((root/f"s{stage}").glob("update_*.jsonl.gz"))))
print(counts,flush=True)

