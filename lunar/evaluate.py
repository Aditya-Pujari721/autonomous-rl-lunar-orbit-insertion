"""evaluate the frozen policy and paired no-thrust baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .capture_nominal import Capture, collect, load_parent, metrics

BENCHMARK = Path(__file__).resolve().parents[1] / "model"

def burn_stats(rows):
    counts = [r['burns'] for r in rows]
    return dict(mean=float(np.mean(counts)), median=float(np.median(counts)),
                minimum=min(counts), maximum=max(counts)) if counts else None

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=512, help='First N fixed confirmation cases')
    parser.add_argument('--output', type=Path, default=Path('runs/benchmark_evaluation'))
    args = parser.parse_args()
    if not 1 <= args.limit <= 512:
        parser.error('--limit must be between 1 and 512')
    manifest = json.loads((BENCHMARK/'manifest.json').read_text())
    digest = hashlib.sha256((BENCHMARK/'policy.pt').read_bytes()).hexdigest()
    if digest != manifest['sha256']:
        raise ValueError('Frozen policy hash mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    actor = load_parent(BENCHMARK/'policy.pt')
    actor.eval()
    seeds = manifest['confirmation'][:args.limit]
    saved = json.loads((BENCHMARK/'confirmation.json').read_text())[:args.limit]
    result = {}
    for name, coast in [('policy', False), ('coast', True)]:
        rows = []
        for start in range(0, len(seeds), 64):
            _, batch = collect(actor, Capture(), seeds[start:start+64], 42, True, coast)
            rows.extend(batch)
            print(name, metrics(rows), flush=True)
        result[name] = rows
        (args.output/f'{name}.json').write_text(json.dumps(rows))
    for expected, actual in zip(saved, result['policy']):
        for key in ['seed', 'verified_orbit', 'event', 'burns']:
            if expected[key] != actual[key]:
                raise AssertionError(f'Frozen regression mismatch: {key}, seed {actual["seed"]}')
        np.testing.assert_allclose(actual['fuel_used_kg'], expected['fuel_used_kg'], rtol=1e-9, atol=1e-6)
    assert all(r['burns'] == 0 and r['fuel_used_kg'] == 0 for r in result['coast'])
    summary = {name: metrics(rows) | {'burns_all': burn_stats(rows),
        'burns_successful': burn_stats([r for r in rows if r['verified_orbit']])}
        for name, rows in result.items()}
    summary.update(checkpoint_sha256=digest, matched_initial_seeds=seeds,
                   frozen_regression_passed=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k != 'matched_initial_seeds'}, indent=2))

if __name__ == '__main__':
    main()
