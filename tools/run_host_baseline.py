"""Run original clyv CPU/GPU pipeline to establish baseline."""
import argparse
import subprocess
import sys
from common import ensure_clyv_path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--video',required=True)
    ap.add_argument('--device',default='cpu')
    ap.add_argument('--show',action='store_true')
    args=ap.parse_args()
    src=ensure_clyv_path()
    cmd=[sys.executable,str(src/'scripts/10_run_pipeline.py'),'--video',args.video,'--device',args.device]
    if args.show: cmd.append('--show')
    print('RUN:',' '.join(cmd),flush=True)
    subprocess.run(cmd,cwd=src,check=True)
if __name__=='__main__':main()
