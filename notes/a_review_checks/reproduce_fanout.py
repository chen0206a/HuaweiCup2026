"""Re-run the hand-built fixture with an unmodified extracted evaluator."""
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--package-root', type=Path, required=True)
args = parser.parse_args()
sys.path.insert(0, str(args.package_root / 'code'))
from evaluation_validation import read_evaluation_config
from multicore_cut_evaluate_problem_1 import evaluate_scene_a
from multicore_cut_evaluate_problem_2 import evaluate_scene_b

fixture = json.loads(Path(__file__).with_name('fanout_check.json').read_text())
config = read_evaluation_config(args.package_root / 'data/config.txt')
results = {}
for scene, fn, waits in [
    ('A', evaluate_scene_a, {'cross_core_wait': 1000, 'same_core_wait': 100}),
    ('B', evaluate_scene_b, {'cross_core_copy_delay': 500}),
]:
    result = fn(fixture['graph'], fixture['plan'], **config, **waits)
    results[scene] = {'makespan': result['makespan'], 'traffic': result['data_movement_bytes']}
assert results == fixture['results'], (results, fixture['results'])
print(json.dumps(results, indent=2))
