import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from lunar.capture_nominal import Capture, load_parent

class BenchmarkTests(unittest.TestCase):
    def test_frozen_integrity_and_inference(self):
        root=Path('model')
        manifest=json.loads((root/'manifest.json').read_text())
        self.assertEqual(hashlib.sha256((root/'policy.pt').read_bytes()).hexdigest(), manifest['sha256'])
        self.assertEqual(len(set(manifest['confirmation'])),512)
        env=Capture();state=env.reset(manifest['confirmation'][0])
        actor=load_parent(root/'policy.pt')
        with torch.no_grad():
            cat,normal,value=actor(torch.tensor(env.observation(state))[None],torch.tensor(env.available(state))[None])
        self.assertTrue(torch.isfinite(normal.mean).all())
        self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(normal.mean.shape,(1,5))
        self.assertEqual(float(cat.probs[0,2]),0.)
        self.assertFalse(state['errors'])

    def test_coast_baseline_matches_confirmation_seeds(self):
        coast=json.loads(Path('results/coast_512.json').read_text())
        policy=json.loads(Path('model/confirmation.json').read_text())
        self.assertEqual([r['seed'] for r in coast],[r['seed'] for r in policy])
        self.assertEqual(sum(r['verified_orbit'] for r in coast),0)
        self.assertTrue(all(r['burns']==0 and r['fuel_used_kg']==0 for r in coast))
