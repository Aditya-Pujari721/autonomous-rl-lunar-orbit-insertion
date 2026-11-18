"""capture-only ppo continuation, exact navigation/execution."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
import torch
from .tasks import TaskEnvironment
from . import environment as base
from .physics import advance, ISP1, MM
from .policy import Actor, likelihood, update


class Capture(TaskEnvironment):
    def __init__(self, collision_cost=25.):
        super().__init__('capture', compress_dry=True)
        self.collision_cost = collision_cost

    def reset(self, seed):
        s = super().reset(seed, errors=False)
        s.update(initial_fuel=s['f1'], verified_orbit=False,
                 max_decisions=float('inf'))
        return s

    def step(self, s, mode, raw):
        before = self.potential(s)
        old_return = s['return_']
        command = np.r_[raw[:3], raw[3] if mode == 1 else raw[4]]
        _, _, term, trunc = super().step(s, mode, command)
        assert not trunc
        if s['success']:
            check = advance(s['r'], s['v'], s['t'], 86400., base.mass(s),
                            0., 0., ISP1, np.zeros(3), s['phase'],
                            accuracy=s.get('accuracy', 1.))
            s['verified_orbit'] = bool(check[5] == 0 and check[-1] >= 86399.)
            s['success'] = s['verified_orbit']
            if not s['success']:
                # resume from the state at the start of the verification coast.
                s['hold'] = 0.
                term = bool(s['event'] or s['t'] >= s['horizon'])
        reward = (self.potential(s)-before + 50*s['verified_orbit']
                  -self.collision_cost*bool(s['event']))
        s['return_'] = old_return+reward
        return float(reward), term

    def summary(self, s):
        r = super().summary(s)
        r.update(verified_orbit=s['verified_orbit'],
                 initial_fuel_kg=s['initial_fuel'], final_fuel_kg=s['f1'],
                 fuel_used_kg=s['initial_fuel']-s['f1'])
        return r


def collect(actor, env, seeds, action_seed, deterministic=False, coast=False):
    torch.manual_seed(action_seed)
    states = [env.reset(int(seed)) for seed in seeds]
    buffers = [[] for _ in states]
    rows = [None]*len(states)
    alive = list(range(len(states)))
    while alive:
        x = np.stack([env.observation(states[i]) for i in alive])
        mask = np.stack([env.available(states[i]) for i in alive])
        with torch.no_grad():
            cat, normal, value = actor(torch.tensor(x), torch.tensor(mask))
            mode = cat.logits.argmax(-1) if deterministic else cat.sample()
            raw = normal.mean if deterministic else normal.sample()
            lp = likelihood(cat, normal, mode, raw)
        remaining = []
        for j, i in enumerate(alive):
            s = states[i]
            start = s['t']
            m = 0 if coast else int(mode[j])
            command = np.array([0.,0.,0.,0.,30.]) if coast else raw[j].numpy()
            reward, done = env.step(s, m, command)
            buffers[i].append((x[j],mask[j],m,command.copy(),float(lp[j]),
                               float(value[j]),reward,s['t']-start))
            if done:
                rows[i] = env.summary(s) | {'seed':int(seeds[i])}
            else:
                remaining.append(i)
        alive = remaining
    return buffers, rows


def metrics(rows):
    successes = [r for r in rows if r['verified_orbit']]
    return dict(n=len(rows), captures=len(successes),
                impacts=sum(bool(r['event']) for r in rows),
                median_fuel_success_kg=float(np.median([r['fuel_used_kg'] for r in successes])) if successes else None,
                mean_reward=float(np.mean([r['reward'] for r in rows])))


def load_parent(path):
    cp = torch.load(path, map_location='cpu', weights_only=False)
    actor = Actor()
    weights = cp['model']
    if weights['control.weight'].shape[0] == 4:
        # exact functional conversion: old shared duration initializes both heads.
        weights = {k:v.clone() for k,v in weights.items()}
        for key in ('control.weight','control.bias','logstd'):
            weights[key] = torch.cat((weights[key],weights[key][3:4]),dim=0)
    actor.load_state_dict(weights)
    return actor


