

import random
import numpy as np

class Svg:

    def __init__(self):
    
        self.out = "";
        
        self.layered = "";
        
        self.semt_minx =0;
        self.semt_miny =0;
        
    def semt(self, x,y):    
        self.semt_minx =x;
        self.semt_miny =y;

#        self.semt_minx, self.semt_miny

    
    def head(self, h, w):
        return '<svg width="' + str(w) + '" height="' + str(h)  + '"  xmlns="http://www.w3.org/2000/svg"> \n ';
        
    def head_auto(self, h, w):
        return '<svg width="' + str(w) + '" height="' + str(h)  + '" viewBox="'+str(-w)+' '+str(-h)+' '+str(2*w)+' '+str(2*h) +'" xmlns="http://www.w3.org/2000/svg"> \n ';

    def head_ext(self, h, w, h2, w2):
        #return '<svg width="' + str(w) + '" height="' + str(h)  + '" viewBox="'+str(self.semt_minx)+' '+str(self.semt_miny)+' '+str(self.semt_minx+w2)+' '+str(self.semt_miny + h2) +'" xmlns="http://www.w3.org/2000/svg"> \n ';
        return '<svg width="' + str(w) + '" height="' + str(h)  + '" viewBox="'+str(0)+' '+str(0)+' '+str(w2)+' '+str(h2) +'" xmlns="http://www.w3.org/2000/svg"> \n ';
                
    def path(self):
        s = '<path d="';
        return s;
    
    def pathend(self, str="black", fl="none"):
        s = '" fill="'+fl+ '" stroke="' + str +'"/>';
        return s;    
    
    def tail(self):
        s = "\n  </svg>";
        return s;
    
    def taglayer(self, h, w):
        self.layered = self.head_auto(h,w) + self.layered + self.tail();
    
    def mabs(self, x,y):
        s = "M"+str(round(x,3))+","+str(round(y,3))+" ";
        return s;
    
    def labs(self, x,y):
        s = "L"+str(round(x,3))+","+str(round(y,3))+" ";
        return s;
    
    def drawcent(self, cn, lasthit = False):        
        s = self.mabs(cn.centre[0], cn.centre[1]+ cn.range[1])
        s += self.labs(cn.centre[0] - cn.range[0], cn.centre[1])
        s += self.labs(cn.centre[0], cn.centre[1]- cn.range[1])
        s += self.labs(cn.centre[0] + cn.range[0], cn.centre[1])
        s += self.labs(cn.centre[0], cn.centre[1]+ cn.range[1])
        if lasthit:
            s += self.mabs(cn.centre[0], cn.centre[1]);
            s += self.labs(cn.lasthit[0], cn.lasthit[1]);        
        return s;
        
    def drawkit(self, k, h, w, lh=False, m2=False, hA=False):
        s = self.head(h,w);
        if(hA):
            s = self.head_auto(h,w);
            
        s += self.path();
        for x in range(0, len(k.centroids)): #k.centroids:
            if(m2):
                k.centroids[x].velocity.centre += k.centroids[x].centre;
                k.centroids[x].lasthit = k.centroids[x].velocity.centre;
            s += self.drawcent(k.centroids[x], lh);
            if(m2):
                s += self.drawcent(k.centroids[x].velocity, False);

        s += self.pathend();
        s += self.tail();
        
        self.out = s;
        return s;
        
    def draw_sem(self, sobj, col): ##Delta?
        s = self.path();
        s += self.mabs(sobj["dl"]["dx"][0], sobj["dl"]["dy"][0])
        for i in range(1, len(sobj["dl"]["dx"])):
            s += self.labs(sobj["dl"]["dx"][i], sobj["dl"]["dy"][i])
        s += self.mabs(sobj["dr"]["dx"][0], sobj["dr"]["dy"][0])
        for i in range(1, len(sobj["dr"]["dx"])):
            s += self.labs(sobj["dr"]["dx"][i], sobj["dr"]["dy"][i])
        s += self.pathend(str=col);
        return s;

    def drawkit_sem(self, xmap, inds, h, w, lh=False, m2=False, hA=False):
        s = self.head(h,w);
        if(hA):
            s = self.head_auto(h,w);            
        #s += self.path();
        #for x in range(0, len(k.centroids)): #k.centroids:
        for iter in inds: #k.centroids:
            #if(m2):
            #    k.centroids[x].velocity.centre += k.centroids[x].centre;
            #    k.centroids[x].lasthit = k.centroids[x].velocity.centre;
            
            s += self.draw_sem(xmap.lane_delta(iter), "#F008");
            #if(m2):
            #    s += self.drawcent(k.centroids[x].velocity, False);

        #s += self.pathend();
        s += self.tail();
        
        self.out = s;
        return s;



    def fileout(self, fn):
        
        f = open(fn + ".svg", "w")
        f.write(self.out);
        f.close();
        
        if len(self.out.split('\n'))>1:
            self.layered += self.out.split('\n')[1];
        
        
    def layerout(self, fn):
        
        f = open(fn + ".svg", "w")
        f.write(self.layered);
        f.close();        
        self.layered = "";
        
        
    def draw_semTraff(self, sobj, col): ##Delta?
        if(len(sobj["px"]) == 0): 
            return "";            
        s = self.path();
        s += self.mabs(sobj["px"][0] - self.semt_minx, sobj["py"][0] - self.semt_miny)
        for i in range(1, len(sobj["px"])):
            s += self.labs(sobj["px"][i] - self.semt_minx, sobj["py"][i] - self.semt_miny)
        
        s += self.pathend(str=col);
        return s;

    def drawkit_semTraff(self, xmap, inds, h, w, lh=False, m2=False, hA=False):
        s = self.head(h,w);
        if(hA):
            s = self.head_auto(h,w);            

        for iter in inds: #k.centroids:
            
            s += self.draw_semTraff(xmap.traffics[iter].Delta[0], "#080A");

        s += self.tail();
        
        self.out = s;
        return s;

## Reimp using mapII

    def draw_semTraffLanes(self, sobj, col): ##Delta?
        if(len(sobj["px"]) == 0): 
            return "";            
        s = self.path();
        s += self.mabs(sobj["px"][0] - self.semt_minx, sobj["py"][0] - self.semt_miny)
        for i in range(1, len(sobj["px"])):
            s += self.labs(sobj["px"][i] - self.semt_minx, sobj["py"][i] - self.semt_miny)
        
        s += self.pathend(str=col);
        return s;

    def drawkit_semTraffLanes(self, xmap, inds, h, w, h2=0, w2=0, lh=False, m2=False, hX=False):
        s = self.head(h,w);
        
        if(hX):
            s = self.head_ext(h,w, h2, w2);            

        for iter in inds: #k.centroids:
            
            s += self.draw_semTraffLanes(xmap.lanes[iter].Delta[0], "#F008");
            
            s += self.draw_semTraffLanes(xmap.lanes[iter].Delta[1], "#00F8");

        s += self.tail();
        
        self.out = s;
        return s;

    
    def draw_vec(self, vec, col):
        
        #def draw_sem(self, sobj, col): ##Delta?
        s = self.path();
        s += self.mabs(vec[0][0], vec[1][0])
        for i in range(1, len(vec[1])):
            s += self.labs(vec[0][i], vec[1][i])
            
        s += self.pathend(str=col);            
        return s;
    

    
    def drawkit_mat(self, mat, defhw = [200,200]):
        
        n = mat.shape;
        
        hscale = defhw[0] / (np.max(mat) - np.min(mat));
        
        nmin = np.min(mat);
        
        wstep = np.arange(0,n[1]) * (defhw[1] / (n[1] - 1));
        
        s = self.head(defhw[0],defhw[1]);
        
        for i in range(0, n[0]):
            
            rc = '#{:0x}'.format(random.randint(256, 16**3-1))
            
            s += self.draw_vec([wstep, (mat[i,:] - nmin) * hscale], rc);
            
        s += self.tail();
        
        self.out = s;
        return s;
        

        
    def drawkit_matII(self, matx, maty, defhw = [200,200], disable_Resize = False):
        
        n = matx.shape;
        
        hscale, vscale, nminx, nminy = 1,1,0,0;
        
        if not disable_Resize:
        
            hscale = defhw[0] / (np.max(matx) - np.min(matx));
        
            nminx = np.min(matx);
        
            vscale = defhw[1] / (np.max(maty) - np.min(maty));
        
            nminy = np.min(maty);
            
                
        s = self.head(defhw[0],defhw[1]);
        
        for i in range(0, n[0]):
            
            rc = '#{:0x}'.format(random.randint(256, 16**3-1))
            
            s += self.draw_vec([(matx[i,:] - nminx) * hscale, (maty[i,:] - nminy) * vscale], rc);
            
        s += self.tail();
        
        self.out = s;
        return s;

    
    def drawkit_line(self, xV, yV, col):
        s = self.path();        
        s += self.mabs(xV[0], yV[0])
        for i in range(1, len(xV)):            
            s += self.labs(xV[i], yV[i])            
        s += self.pathend(str=col);     
        #self.out = s;
        return s;    



class quickcent:
    
    centre = [];
    range = [];

    def __init__(self, centre, xrange):
        self.centre = centre;
        self.range = xrange;
    
class quickkit:
    centroids = [];
    
