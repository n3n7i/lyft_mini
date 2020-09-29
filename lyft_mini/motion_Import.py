

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numba
        
import l5kit.geometry as l5geo
r33yaw = l5geo.rotation33_as_yaw;

#import zarr

#import sys, os
#from pathlib import Path

#sys.path.append("../input/pycent/");
#import pyCent as pkit

#sys.path.append("../input/pysvg/");
#import pySvg as gkit

#ptime("import loading: ", st);

#timeDEBUG = False;

def recordAccess(data, sc):
    rS = data["scenes"][sc];    
    rF = data["frames"][slice(rS[0][0], rS[0][1])];    
    rA = data["agents"][slice(rF[0][1][0], rF[-1][1][1])];
    return [rS, rF, rA];

@numba.stencil()
def filter1d(data):
    return (data[0] - data[-10]); ## ~1 sec delta

@numba.stencil()
def filter1d4(data):
    return (data[0] - data[-4]);


def translations(rf):
    
    ego_x = rf["ego_translation"][:, 0]
    ego_y = rf["ego_translation"][:, 1]
    
    #print(rf["ego_translation"].shape)
    
    #etx = filter1d(ego_x);
    #ety = filter1d(ego_y);
    
    return [ego_x, ego_y]; #[etx, ety];


#r33yaw = l5geo.rotation33_as_yaw;


def collectYaw(rf):
    
    ego_R = rf["ego_rotation"][:][:,:]
    
    #print(rf["ego_rotation"].shape)
    
    rvec = np.zeros(ego_R.shape[0]);
    for i in range(0, ego_R.shape[0]):
        rvec[i] = r33yaw(ego_R[i]);
    return rvec;


def rotations(r):
    etR = filter1d(r);
    return etR;


def frameAccess(data, sc):
    rS = data["scenes"][sc];    
    #rF = data["frames"][slice(rS[0][0], rS[0][1])]; 
    rF = data.frames[slice(rS[0][0], rS[0][1])]; 
    return [rS, rF];


def frame_Data(scene, rf):
    
   # if timeDEBUG: ptime("fDInit: ", st);
    #rf = frameAccess(zdz, scene)[1];
    
    yaws = collectYaw(rf); # heading
    
  #  if timeDEBUG: ptime("fDYaw: ", st);
    
    t = translations(rf); # vels    
    #r = rotations(yaws); # heading'
    #v1d = np.hypot(t[0], t[1]); # combined vel
 #   if timeDEBUG: ptime("fDUpper: ", st);
    #accel = filter1d(v1d[10:]); #  vel'
    #rad_cel = filter1d(r[10:]); # heading''
#    if timeDEBUG: ptime("fD: ", st);
    
    #vr = np.array([v1d[20:], accel[10:], r[20:], rad_cel[10:]]).T[:200, :];
    
    
    
 #   if timeDEBUG: ptime("fDEnd: ", st);
    
    #vr = np.array([t, yaws]).T;
    vr = np.vstack([t, yaws]);
    return vr;
    

def collectFrames(xr, zdz):

    tx = frameAccessII(zdz, [xr[0], xr[-1]+1]);
    
    xlist = [];
    splist = [];
    spl =0;
    for i in xr:
        
        if(i%500 == 1): print(" step ",i, end='');
        
        tn = frameAccessIII(tx, i);
        
        ##print(tn.shape)
        
        xlist.append(frame_Data(i,tn));
        
        spl += tn.shape[0];
        splist.append(spl);
        
        
    print(" ");
    
    return [xlist, splist];

def frameAccessII(data, sc):
    rS = data.scenes[slice(sc[0], sc[1])];
  #  if timeDEBUG: ptime("faII: ", st);
    
    #rF = data["frames"][slice(rS[0][0], rS[0][1])]; 
    rF = data.frames[slice(rS[0][0][0], rS[-1][0][1])];
  #  if timeDEBUG: ptime("faIIB: ", st);
    
    
    return [rS, rF];

def frameAccessIII(rx, sc):
    
    #print(len(rx))
    rS = rx[0][sc];    
    #rF = data["frames"][slice(rS[0][0], rS[0][1])]; 
    ##eSlice = np.min(rS[0][1] - rS[0][0], 240);
    
    rF = rx[1][slice(rS[0][0], rS[0][1])];     
  #  if timeDEBUG: ptime("fAIIIB ", st);
    return rF;

