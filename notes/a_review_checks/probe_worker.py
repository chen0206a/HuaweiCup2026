import sys,json,time,argparse
from pathlib import Path
parser=argparse.ArgumentParser(description='Official singlecore timing probe.')
parser.add_argument('graph',type=Path)
parser.add_argument('--package-root',type=Path,required=True)
args=parser.parse_args()
sys.path.insert(0,str(args.package_root/'code'))
from evaluation_validation import read_evaluation_config
from singlecore_evaluate import evaluate_singlecore
p=args.package_root
cfg=read_evaluation_config(p/'data/config.txt')
g=json.loads(args.graph.read_text(encoding='utf-8'))
t=time.perf_counter();r=evaluate_singlecore(g,**cfg)
print(json.dumps({'seconds':time.perf_counter()-t,'makespan':r['makespan'],'added_copy_bytes':r['data_movement_bytes']['added_copy_bytes']}))
