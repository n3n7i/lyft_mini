

import numpy as np

import l5kit.geometry as l5geo
import numba;
r33yaw = l5geo.rotation33_as_yaw;

import l5kit.data.labels as l5Labels

l5Agent = l5Labels.PERCEPTION_LABEL_TO_INDEX;
l5short = [l5Labels.PERCEPTION_LABELS[i][17:] for i in range(0, len(l5Labels.PERCEPTION_LABELS))]

def l5ind(str):
    return l5Agent[("PERCEPTION_LABEL_"+str).upper()]


class data: 
  class ego:
    @numba.stencil()
    def filter1d(data):
      return (data[0] - data[-10]);
  
    def translations(rf):    
      ego_x = rf["ego_translation"][:, 0]
      ego_y = rf["ego_translation"][:, 1]
      return [ego_x, ego_y]; 

    def collectYaw(rf):    
      ego_R = rf["ego_rotation"][:][:,:]
      rvec = np.zeros(ego_R.shape[0]);
      for i in range(0, ego_R.shape[0]):
        rvec[i] = r33yaw(ego_R[i]);
      return rvec;

  def frameAccessII(data, sc):
    rS = data.scenes[slice(sc[0], sc[1])];
    rF = data.frames[slice(rS[0][0][0], rS[-1][0][1])];
    return [rS, rF];

  def frameAccessIII(rx, sc):
    rS = rx[0][sc];    
    rF = rx[1][slice(rS[0][0], rS[0][1])];     
    return rF;

  pass;
  #pass;

class dataFun(data):
  pass;

class dataFun(dataFun):
  
  def frame_Data(scene, rf, self=dataFun):        
    yaws = self.ego.collectYaw(rf); # heading
    t = self.ego.translations(rf); # vels    
    vr = np.vstack([t, yaws]);
    return vr;
  
  pass;

    
class dataFun(dataFun): ##N/A

    
  def yawTest1(self=dataFun):
    return self.ego.filter1d    
        
  def frameAccess(data, sc):#, self=dataFun):        
    rS = data["scenes"][sc];    
    rF = data.frames[slice(rS[0][0], rS[0][1])]; 
    return [rS, rF];

    
  def collectFrames(xr, zdz, self=dataFun):
    tx = self.frameAccessII(zdz, [xr[0], xr[-1]+1]);
    xlist = [];
    splist = [];
    spl =0;
    for i in xr:        
      if(i%500 == 1): print(" step ",i, end='');
      tn = self.frameAccessIII(tx, i);
      xlist.append(self.frame_Data(i,tn));
      spl += tn.shape[0];
      splist.append(spl);        
    print('\n');  
    return [xlist, splist];

##

  """def scene_Assign(subTimes, dataTimes):
    
    ids = np.zeros(subTimes.shape[0], dtype=int) -1;   
    idl = dataTimes.shape[1];
    idx = 0;    
    idR = subTimes.shape[0]; 
    
    print(idR, idl, dataTimes.shape[1], subTimes.shape[0])
    
    print(dataTimes.shape, subTimes.shape)
    
    for xiter in range(0,idR):
      print(xiter, idx);
      if (idx == idl): break;  
      while dataTimes[0, idx] <= subTimes[xiter] <= dataTimes[1, idx]:        
        ids[idx] = xiter;            
        idx += 1;            
        if (idx == idl): break;    
    return ids;                    
#"""
    

  def scene_Assign(subTimes, dataTimes):    
    ids = np.zeros(subTimes.shape[0], dtype=int) -1;   
    idR = dataTimes.shape[1];
    idx = 0;
    idl = subTimes.shape[0];
    for xiter in range(0,idR):
        while dataTimes[0, xiter] <= subTimes[idx] <= dataTimes[1, xiter]:
            ids[idx] = xiter;
            idx += 1;
            if (idx == idl): break;
        if (idx == idl): break;
    return ids;        


  def scene_Targets(frameids, trackids):    
    xlist = [];    
    for i in range(0, np.max(frameids)+1):        
      mask = np.nonzero(frameids == i)[0];        
      xlist.append(trackids[mask]);        
    return xlist;



  def target_AgentsII(data, sc, aglist):
    
    rS = data["scenes"][sc];    
    rF = data["frames"][slice(rS[0][0], rS[0][1])]; ##?
    rA = data["agents"][slice(rF[0][1][0], rF[-1][1][1])];
    idR = rA["track_id"]
    asLanes = [];
    for x in aglist:
      asLanes.append(rA[np.nonzero(idR==x)[0]])
    return asLanes;


  def csvTrainer(data, sc, targType, ts = [1]):
    
    csv = [];    
    for sci in sc:
      rS = data["scenes"][sci];
      rF = data["frames"][slice(rS[0][0], rS[0][1])];      
      for rz in ts:
        rA = data["agents"][slice(rF[rz][1][0], rF[rz][1][1])];        
        ts2 = rF["timestamp"][rz];
        for agi in rA:
          if(agi["label_probabilities"][targType] == 1):
            tl = agi["track_id"];
            csv.append([ts2, tl]);
    return csv;

