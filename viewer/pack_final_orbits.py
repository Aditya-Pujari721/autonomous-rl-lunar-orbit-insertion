import gzip,struct,json,glob,math
from pathlib import Path
import numpy as np
from lunar.physics import moon,MM,RM,INC,OMEGA
b=gzip.open('viewer/static/training-v2.bin.gz','rb').read(); idx=0; flags=[]
while idx<len(b):
 stage,n,hit=b[idx:idx+3];flags.append(hit);idx+=3+n*16
assert idx==len(b) and len(flags)==18151
rows={}
for f in sorted(glob.glob('runs/recorded_capture/s3/update_*.gz')):
 group={}
 for line in gzip.open(f,'rt'):
  r=json.loads(line);group[r[0]]=r
 rows.update(group)
assert len(rows)==1024
out={}
for k,row in enumerate(rows.values(),len(flags)-1024):
 if k<99*len(flags)//100 or not flags[k]:continue
 t=row[12]; phase=row[2];j,jv=moon(t,phase);r=np.array(row[13:16])-j;v=np.array(row[16:19])-jv
 a=1/(2/np.linalg.norm(r)-v@v/MM);h=np.cross(r,v);ev=np.cross(v,h)/MM-r/np.linalg.norm(r);e=np.linalg.norm(ev)
 if not(0<=e<1 and a*(1-e)>RM):continue
 p=ev/e;q=np.cross(h/np.linalg.norm(h),p)
 E=math.atan2(r@q/(a*np.sqrt(1-e*e)),r@p/a+e);ang=phase+t*OMEGA
 def basis(vec):
  x,y,z=vec;yp=y*np.cos(INC)+z*np.sin(INC);return [x*np.cos(ang)+yp*np.sin(ang),-x*np.sin(ang)+yp*np.cos(ang),z*np.cos(INC)-y*np.sin(INC)]
 out[k]=[a/RM,e,E-e*np.sin(E),basis(p),basis(q)]
assert out
Path('viewer/static/training-orbits.json').write_text(json.dumps(out))
print(len(flags),len(out))
