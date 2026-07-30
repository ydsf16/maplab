#!/usr/bin/env python3
"""Create per-mission DA3 windows from a joint multi-session result."""
import argparse, csv, json, sys
from pathlib import Path
import cv2, numpy as np

GEOM = Path(__file__).resolve().parents[1] / "sensor-recorder-geometry"
sys.path.insert(0, str(GEOM))
from prepare_da3_input import pose, angle

def main():
 p=argparse.ArgumentParser(); p.add_argument('--joint-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--window-size',type=int,default=20); p.add_argument('--window-overlap',type=int,default=4); a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True); sessions=json.loads((a.joint_output/'sessions.json').read_text()); manifest={'coordinate':'Maplab RIGHT_HAND_Z_UP; DA3 extrinsic T_C_M','sessions':[],'windows':[]}
 for s in sessions:
  sid=s['id']; result=Path(s['result']); raw=Path(s['raw_data'])
  frames=list(csv.DictReader((result/'normalized/frames.csv').open())); tum=[pose(x) for x in (a.joint_output/'poses'/sid/'image_poses_tum.txt').read_text().splitlines() if x.strip()]; ts=np.array([x[0] for x in tum]); poses=[x[1] for x in tum]
  selected=[]; last=None
  for row in frames:
   t=float(row['timestamp_ns'])*1e-9; idx=int(np.argmin(abs(ts-t)))
   if abs(ts[idx]-t)>.02: continue
   cur=poses[idx]
   if last is None or np.linalg.norm(cur[:3,3]-last[1][:3,3])>=.15 or angle(last[1],cur)>=5 or t-last[0]>=1.0: selected.append((row,t,cur)); last=(t,cur)
  report=json.loads((result/'maps/08_visual_inertial_ba_loops_preview/report.json').read_text()); intr=report['calibration']['final_intrinsics']; manifest['sessions'].append({'id':sid,'result':str(result),'raw_data':str(raw),'frames':len(selected),'intrinsics_source':'single-session final VI-BA report.json'})
  for start in range(0,len(selected),a.window_size-a.window_overlap):
   part=selected[start:min(start+a.window_size,len(selected))]
   if not part: break
   out=a.output/'windows'/sid/f'window_{start//(a.window_size-a.window_overlap):03d}'/'input'; images=out/'images'; images.mkdir(parents=True,exist_ok=True); cap=cv2.VideoCapture(str(raw/'wide.mp4')); K=[]; E=[]; rows=[]
   for j,(row,t,T) in enumerate(part):
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(row['record_slot'])); ok,img=cap.read()
    if not ok: raise RuntimeError(f'cannot decode {sid} slot {row["record_slot"]}')
    name=f'frame_{j:06d}.jpg'; cv2.imwrite(str(images/name),img,[cv2.IMWRITE_JPEG_QUALITY,95]); sx=img.shape[1]/float(row['width_px']); sy=img.shape[0]/float(row['height_px']); K.append([[intr[0]*sx,0,intr[2]*sx],[0,intr[1]*sy,intr[3]*sy],[0,0,1]]); E.append(np.linalg.inv(T)); rows.append({'session_id':sid,'timestamp_ns':row['timestamp_ns'],'record_slot':row['record_slot'],'image':f'images/{name}'})
   cap.release(); np.savez_compressed(out/'camera_params.npz',intrinsics=np.asarray(K),extrinsics=np.asarray(E));
   with (out/'frames.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
   manifest['windows'].append({'session_id':sid,'input':str(out),'frames':len(part)})
   if start+a.window_size>=len(selected): break
 (a.output/'joint_camera_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
if __name__=='__main__': main()
