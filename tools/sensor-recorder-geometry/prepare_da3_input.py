#!/usr/bin/env python3
"""Prepare pose-conditioned DA3 input from a final single-session VI-BA."""
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np

def pose(line):
    v=np.array([float(x) for x in line.split()]); qx,qy,qz,qw=v[4:];
    R=np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],[2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],[2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
    T=np.eye(4); T[:3,:3]=R; T[:3,3]=v[1:4]; return v[0],T
def angle(A,B): return np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1)))
def main():
 p=argparse.ArgumentParser(); p.add_argument('--slam-output',type=Path,required=True); p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--min-translation-m',type=float,default=.15); p.add_argument('--min-rotation-deg',type=float,default=5); p.add_argument('--max-interval-s',type=float,default=1.0); p.add_argument('--max-frames',type=int,default=0); p.add_argument('--start-index',type=int,default=0); p.add_argument('--end-index',type=int,default=-1); p.add_argument('--metadata-only',action='store_true'); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
 frames=list(csv.DictReader((a.slam_output/'normalized/frames.csv').open())); tum=[pose(x) for x in (a.slam_output/'poses/image_poses_tum.txt').read_text().splitlines() if x.strip()]; ts=np.array([x[0] for x in tum]); poses=[x[1] for x in tum]
 selected=[]; last=None
 for row in frames:
  t=float(row['timestamp_ns'])*1e-9; i=int(np.argmin(abs(ts-t)))
  if abs(ts[i]-t)>.02: continue
  cur=poses[i]
  if last is None or np.linalg.norm(cur[:3,3]-last[1][:3,3])>=a.min_translation_m or angle(last[1],cur)>=a.min_rotation_deg or t-last[0]>=a.max_interval_s:
   selected.append((row,t,cur)); last=(t,cur)
 if a.max_frames > 0 and len(selected)>a.max_frames:
  selected=[selected[i] for i in np.linspace(0,len(selected)-1,a.max_frames,dtype=int)]
 end=len(selected) if a.end_index < 0 else min(a.end_index,len(selected))
 selected=selected[a.start_index:end]
 if not selected: raise RuntimeError('Selected DA3 frame interval is empty')
 if a.metadata_only:
  (a.output/'manifest.json').write_text(json.dumps({'frames':len(selected),'selected_frame_range':[a.start_index,end]},indent=2)+'\n')
  return
 cap=cv2.VideoCapture(str(a.data/'wide.mp4')); images=a.output/'images'; images.mkdir(exist_ok=True); K=[]; E=[]; out=[]
 report=json.loads((a.slam_output/'maps/08_visual_inertial_ba_loops_preview/report.json').read_text()); intr=report['calibration']['final_intrinsics']
 for j,(row,t,T) in enumerate(selected):
  cap.set(cv2.CAP_PROP_POS_FRAMES,int(row['record_slot'])); ok,img=cap.read()
  if not ok: raise RuntimeError('cannot decode record_slot '+row['record_slot'])
  name=f'frame_{j:06d}.jpg'; cv2.imwrite(str(images/name),img,[cv2.IMWRITE_JPEG_QUALITY,95])
  sx=img.shape[1]/float(row['width_px']); sy=img.shape[0]/float(row['height_px'])
  k=np.array([[intr[0]*sx,0,intr[2]*sx],[0,intr[1]*sy,intr[3]*sy],[0,0,1]],float); K.append(k); E.append(np.linalg.inv(T)); out.append({'da3_index':j,'source_frame_index':int(row['frame_index']),'record_slot':int(row['record_slot']),'timestamp_ns':int(row['timestamp_ns']),'image':f'images/{name}','source_width_px':img.shape[1],'source_height_px':img.shape[0],'intrinsics_scale_x':sx,'intrinsics_scale_y':sy})
 cap.release(); np.savez_compressed(a.output/'camera_params.npz',intrinsics=np.array(K),extrinsics=np.array(E));
 with (a.output/'frames.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=out[0]); w.writeheader(); w.writerows(out)
 (a.output/'manifest.json').write_text(json.dumps({'frames':len(out),'selected_frame_range':[a.start_index,end],'pose':'T_M_C from final VI-BA image_poses_tum.txt, exported to DA3 as T_C_M','intrinsics':'final_intrinsics from maps/08_visual_inertial_ba_loops_preview/report.json, scaled from SLAM resolution to source video resolution','units':'meters','camera':'OpenCV RDF'},indent=2)+'\n')
if __name__=='__main__': main()
