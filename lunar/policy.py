"""neural actor and unchanged ppo update used by the frozen capture policy."""
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(25,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
        self.mode=nn.Linear(128,3);self.control=nn.Linear(128,5)
        self.value=nn.Linear(128,1);self.logstd=nn.Parameter(torch.zeros(5))
        for m in self.modules():
            if isinstance(m,nn.Linear):nn.init.orthogonal_(m.weight,np.sqrt(2));nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mode.weight,.01);nn.init.orthogonal_(self.control.weight,.01)
        nn.init.orthogonal_(self.value.weight,1.)
    def forward(self,x,mask):
        h=self.body(x)
        return (Categorical(logits=self.mode(h).masked_fill(~mask,-1e9)),
                Normal(self.control(h),self.logstd.clamp(-2,1).exp()),self.value(h).squeeze(-1))

def active(mode):
    return torch.stack((mode==1,mode==1,mode==1,mode==1,mode==0),dim=-1)

def likelihood(cat,normal,mode,raw):
    return cat.log_prob(mode)+(normal.log_prob(raw)*active(mode)).sum(-1)
def targets(reward,value,elapsed,trace_seconds):
    """gamma=1; time-based trace decay; critic target is full-flight return."""
    adv=np.zeros(len(reward));returns=np.zeros(len(reward));tail=0.;total=0.
    for k in reversed(range(len(reward))):
        nv=value[k+1] if k+1<len(value) else 0.
        total+=reward[k];returns[k]=total
        tail=reward[k]+nv-value[k]+np.exp(-elapsed[k]/trace_seconds)*tail
        adv[k]=tail
    return adv,returns

def update(actor,opt,buffers,trace_seconds):
    data=[];advantages=[];returns=[]
    for b in buffers:
        a,r=targets([x[6] for x in b],[x[5] for x in b],[x[7] for x in b],trace_seconds)
        data+=b;advantages.extend(a);returns.extend(r)
    x=torch.tensor(np.stack([r[0] for r in data]));mask=torch.tensor(np.stack([r[1] for r in data]))
    modes=torch.tensor([r[2] for r in data]);raw=torch.tensor(np.stack([r[3] for r in data]))
    old=torch.tensor([r[4] for r in data]);a=torch.tensor(advantages,dtype=torch.float32)
    a=(a-a.mean())/a.std(unbiased=False).clamp(min=1e-8)
    ret=torch.tensor(returns,dtype=torch.float32);kls=[]
    for epoch in range(4):
        ek=[]
        for ix in torch.randperm(len(x)).split(256):
            c,d,v=actor(x[ix],mask[ix]);logratio=likelihood(c,d,modes[ix],raw[ix])-old[ix]
            ratio=logratio.exp();loss=-torch.minimum(ratio*a[ix],ratio.clamp(.8,1.2)*a[ix]).mean()
            entropy=(c.entropy()+c.probs[:,1]*d.entropy()[:,:4].sum(-1)
                     +c.probs[:,0]*d.entropy()[:,4]).mean()
            loss=loss+.5*(v-ret[ix]).square().mean()-.01*entropy
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(actor.parameters(),.5);opt.step()
            ek.append(float(((ratio-1)-logratio).mean().detach()))
        kls.extend(ek)
        if np.mean(ek)>.03:break
    return float(np.mean(kls))
