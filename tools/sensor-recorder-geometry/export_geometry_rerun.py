#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import rerun as rr
def read_binary_vertex_ply(path):
    types={'char':'i1','uchar':'u1','short':'<i2','ushort':'<u2','int':'<i4','uint':'<u4','float':'<f4','double':'<f8'}; props=[]; n=None
    with path.open('rb') as f:
        while True:
            s=f.readline().decode().strip()
            if s.startswith('element vertex '): n=int(s.split()[-1])
            elif s.startswith('property ') and n is not None and s.split()[1] != 'list': props.append((s.split()[2],types[s.split()[1]]))
            elif s=='end_header': break
        a=np.fromfile(f,dtype=np.dtype(props),count=n)
    p=np.column_stack([a['x'],a['y'],a['z']]).astype(np.float32); c=np.column_stack([a['red'],a['green'],a['blue']]).astype(np.uint8) if all(k in a.dtype.names for k in ('red','green','blue')) else None; return p,c
def main():
 p=argparse.ArgumentParser(); p.add_argument('--ply',type=Path,required=True); p.add_argument('--camera',type=Path,action='append',required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 points,colors=read_binary_vertex_ply(a.ply); c2w=np.concatenate([np.linalg.inv(np.load(camera)['extrinsics']) for camera in a.camera]); a.output.parent.mkdir(parents=True,exist_ok=True)
 rr.init('PhoneAI_DA3_TSDF',spawn=False); rr.save(str(a.output)); rr.log('world',rr.ViewCoordinates.RIGHT_HAND_Z_UP,static=True); rr.log('world/tsdf/pointcloud',rr.Points3D(points,colors=colors,radii=.004),static=True); centers=c2w[:,:3,3]; rr.log('world/final_vi_ba_trajectory',rr.LineStrips3D([centers],colors=[[40,180,255]],radii=[.02]),static=True); rr.log('world/da3_geometry_frames',rr.Points3D(centers,colors=[[255,190,60]],radii=.04),static=True); rr.log('metadata',rr.TextDocument(f'points={len(points)}\nframes={len(centers)}\nworld=Maplab right-hand Z-up, gravity=-Z\npose=T_M_C final VI-BA, meters\nDA3 extrinsic=T_C_M'),static=True)
if __name__=='__main__': main()
