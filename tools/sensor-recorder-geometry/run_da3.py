#!/usr/bin/env python3
import argparse,json,sys,time
from pathlib import Path
import numpy as np, torch
def main():
 p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--repo',type=Path,default=Path('/root/autodl-tmp/da3/repo')); p.add_argument('--model',type=Path,default=Path('/root/autodl-tmp/da3/models/DA3-GIANT-1.1')); p.add_argument('--process-res',type=int,default=504); a=p.parse_args(); sys.path.insert(0,str(a.repo/'src')); from depth_anything_3.api import DepthAnything3
 d=np.load(a.input/'camera_params.npz'); imgs=[str(x) for x in sorted((a.input/'images').glob('*.jpg'))]; a.output.mkdir(parents=True,exist_ok=True); t=time.perf_counter(); model=DepthAnything3.from_pretrained(str(a.model)).to('cuda'); load=time.perf_counter()-t; t=time.perf_counter(); model.inference(imgs,extrinsics=d['extrinsics'],intrinsics=d['intrinsics'],align_to_input_ext_scale=True,process_res=a.process_res,process_res_method='upper_bound_resize',export_dir=str(a.output),export_format='mini_npz'); torch.cuda.synchronize(); stats={'frames':len(imgs),'process_res':a.process_res,'model_load_seconds':load,'inference_seconds':time.perf_counter()-t,'model':str(a.model),'known_pose':True}; (a.output/'run_stats.json').write_text(json.dumps(stats,indent=2)+'\n')
if __name__=='__main__': main()
