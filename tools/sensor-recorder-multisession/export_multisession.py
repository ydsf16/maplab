#!/usr/bin/env python3
import csv, json, os, shutil, sys
from pathlib import Path

_rerun_python = "/usr/bin/python3"
if os.path.isfile(_rerun_python) and os.environ.get("PHONE_AI_RERUN_REEXEC") != "1" and os.path.realpath(sys.executable) != _rerun_python:
    os.environ["PHONE_AI_RERUN_REEXEC"] = "1"
    os.execv(_rerun_python, [_rerun_python, __file__, *sys.argv[1:]])

import numpy as np
import rerun as rr

COLORS = [[52,152,219],[46,204,113],[241,196,15],[231,76,60],[155,89,182]]
def rows(p):
    with Path(p).open() as f: return list(csv.DictReader(f))
def qmat(r):
    w,x,y,z=[float(r[k]) for k in ('q_w','q_x','q_y','q_z')]; n=(w*w+x*x+y*y+z*z)**.5; w,x,y,z=[v/n for v in (w,x,y,z)]
    T=np.eye(4); T[:3,:3]=[[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]]; T[:3,3]=[float(r[k]) for k in ('p_x_m','p_y_m','p_z_m')]; return T
def tum_transform(src,dst,T):
    out=[]
    for l in Path(src).read_text().splitlines():
        v=l.split()
        if len(v)!=8: continue
        t=np.eye(4); x,y,z,qx,qy,qz,qw=map(float,v[1:]); t[:3,3]=[x,y,z]
        r={'q_w':qw,'q_x':qx,'q_y':qy,'q_z':qz,'p_x_m':x,'p_y_m':y,'p_z_m':z}; t[:3,:3]=qmat(r)[:3,:3]; t=T@t
        R=t[:3,:3]; qw=np.sqrt(max(0,1+np.trace(R)))/2; qx=(R[2,1]-R[1,2])/(4*qw); qy=(R[0,2]-R[2,0])/(4*qw); qz=(R[1,0]-R[0,1])/(4*qw)
        out.append(f'{v[0]} {t[0,3]:.9f} {t[1,3]:.9f} {t[2,3]:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}')
    Path(dst).write_text('\n'.join(out)+'\n')
def main():
    import argparse; a=argparse.ArgumentParser(); a.add_argument('--root',type=Path,required=True); ns=a.parse_args(); root=ns.root
    final=rows(root/'24_multisession_vi_ba/export/vertices.csv'); byvid={r['vertex_id']:r for r in final}; bymission={}
    for r in final: bymission.setdefault(r['mission_id'],[]).append(r)
    poses=root/'poses'; poses.mkdir(exist_ok=True); rr.init('sensor_recorder_multisession',spawn=False); rr.save(str(root/'rerun_multisession.rrd'))
    for i,(mission,vs) in enumerate(bymission.items()):
        p=np.array([[float(r[k]) for k in ('p_x_m','p_y_m','p_z_m')] for r in vs]); c=COLORS[i%len(COLORS)]
        rr.log(f'world/trajectories/session_{i}',rr.LineStrips3D([p],colors=[c],radii=[0.03]))
        rr.log(f'world/keyframes/session_{i}',rr.Points3D(p,colors=[c],radii=0.05))
    lm=rows(root/'24_multisession_vi_ba/export/landmarks.csv'); p=np.array([[float(r[k]) for k in ('x_m','y_m','z_m')] for r in lm]); rr.log('world/landmarks',rr.Points3D(p,colors=[255,180,30],radii=0.012))
    sessions=json.loads((root/'sessions.json').read_text())
    for i,s in enumerate(sessions):
        local=rows(Path(s['result'])/'maps/08_visual_inertial_ba_loops_preview/export/vertices.csv')
        if not local: continue
        anchor=next(x for x in local if x['vertex_id'] in byvid); T=qmat(byvid[anchor['vertex_id']])@np.linalg.inv(qmat(anchor))
        out=poses/s['id']; out.mkdir(exist_ok=True)
        tum_transform(Path(s['result'])/'poses/imu_poses_tum.txt',out/'imu_poses_tum.txt',T)
        tum_transform(Path(s['result'])/'poses/image_poses_tum.txt',out/'image_poses_tum.txt',T)
    (poses/'README.txt').write_text('Each session folder contains final joint-map T_M_I and T_M_C TUM poses. Timestamps remain session-local and are therefore not concatenated.\n')
if __name__=='__main__': main()
