


import numpy as np # linear algebra
#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numba

class liveCent:

    #centre;
    #range;
    #lasthit;

    def initCent(self, xdata):
        return np.array([xdata[0], xdata[1]]);
    
    def __init__(self, xdata):
        self.centre = self.initCent(xdata);
        self.range = self.initCent([0.0, 0.0]);
        self.lasthit = self.initCent(xdata);
        
    def __repr__(self):
        r = "lC+[@" + ",".join([str(self.centre), str(self.range)]) + "r] ";
        return r;

    def dist(self, point):        
        #print("s: ", self, ", p: ", point);
        return abs(self.centre[0] - point[0]) + abs(self.centre[1] - point[1]);
    
    #@numba.jit("float32(voidptr, float32, float32)")
    def dist_v(self, pointa, pointb):
        #print("s: ", self, ", p: ", point);       
        #return abs(self.centre[0] - pointa) + abs(self.centre[1] - pointb);
        return np.abs(self.centre[0] - pointa) + np.abs(self.centre[1] - pointb);

        
    def thres(self, point, val):
        return self.dist(point) <= val;
    
    def inrange(self, point):
        return (abs(self.centre[0] - point[0]) < self.range[0]) & (abs(self.centre[1] - point[1]) < self.range[1]);
    
    def inc(self, point):
        rx = abs(self.centre[0] - point[0]) - self.range[0];
        ry = abs(self.centre[1] - point[1]) - self.range[1];
    
        self.lasthit[0] = point[0];
        self.lasthit[1] = point[1];
        
        #print(rx," ", ry);
        if((rx>0) | (ry>0)):
            rx = rx * (rx>0);
            ry = ry * (ry>0);
    
            self.range[0] += (rx / 2);
            self.range[1] += (ry / 2);
            
            self.centre[0] += np.sign(point[0] - self.centre[0]) * (rx / 2);
            self.centre[1] += np.sign(point[1] - self.centre[1]) * (rx / 2);
            
           # print(self.range[0]," ", self.range[1]);
            
        #return self;
        
    def inc_Multi(self, points):
        xmax = max(points[0,:]);
        ymax = max(points[1,:]);
        xmin = min(points[0,:]);
        ymin = min(points[1,:]);
        self.inc([xmax, ymax]);
        self.inc([xmin, ymin]);
        

class liveCentII(liveCent):
    
    #centre;
    #range;
    #velocity :: livecent?    
    
    def __init__(self, xdata, xvel):
        self.centre = self.initCent(xdata);
        self.range = self.initCent([0.0, 0.0]);
        self.lasthit = self.initCent(xdata);
        self.velocity = liveCent(xvel);            

        
class centKit:
    
    #data;
    #centroids;
    #distances;
    #step    
    
    
    def initData(self, xdata):
        return np.asarray(xdata, np.float32);##array([np.asarray(xdata[0]), np.asarray(xdata[1])]);
    
    def pushCent(self):
        self.centroids.append(liveCent([self.data[0][self.step], self.data[1][self.step]]));
    
    def __init__(self, xdata):
        self.data = self.initData(xdata);
        self.step = 0;
        self.centroids = [];
        self.pushCent();        
        
        self.vstepf = np.vectorize(liveCent.dist_v); ##, excluded=['pointa', 'pointb']);
        
        
    def stepf(self):    
        z = len(self.centroids);
        n = 1e5;
        id=-1;
        for i in range(0,z):
            xn = self.centroids[i].dist([self.data[0][self.step], self.data[1][self.step]]);
            if(xn<n):
                n = xn;
                id = i;
        
        return [id, n];    
        #print(n, " ", id);
        
    #def vstepf_init(self):
    #    return np.vectorize(liveCent.dist_v); ##, excluded=['pointa', 'pointb']);
        
    def stepi(self):
        self.step += 1;

        
    #@numba.jit
    def stepPass(self, batch, thres, maxCent):
        c = len(self.centroids);
        for self.step in batch:
            d = [self.data[0][self.step], self.data[1][self.step]];
            r = self.stepf();
            if(r[1] < thres):
                self.centroids[r[0]].inc(d);
                continue;
            if(c < maxCent):
                self.pushCent();
                c+=1;
                continue;
            self.centroids[r[0]].inc(d);
        
        pass;
    
    #@numba.jit
    def stepPassII(self, batch, thres, maxCent):
        c = len(self.centroids);
        for self.step in batch:
            d = [self.data[0][self.step], self.data[1][self.step]];
            r2 = self.vstepf(self.centroids, self.data[0][self.step], self.data[1][self.step]);
            m2 = np.argmin(r2);
            v2 = r2[m2];
        
            if((v2 < thres) | (c>=maxCent)):
                self.centroids[m2].inc(d);
                continue;
            if(c < maxCent):
                self.pushCent();
                c+=1;
                continue;
            #self.centroids[r[0]].inc(d);
        
        return r2;
    
    #@numba.jit
    def stepPassIII(self, batch, thres, maxCent):        
        c = len(self.centroids);
        pT = 1e5;
        
        cn = 0;
        #for self.step in batch:
        while (pT > thres) & (cn < maxCent):
            
            #d = [self.data[0][self.step], self.data[1][self.step]];
            r2 = self.vstepf(self.centroids[cn], self.data[0], self.data[1]);            
            
            m2 = np.argmax(r2);
            v2 = r2[m2];
            
            if v2<pT:
                pT = v2;
            
            cn += 1;
            if((pT > thres) & (cn ==c)):
                self.step = m2;
                self.pushCent();
                c+=1;
        
            #if((v2 < thres) | (c>=maxCent)):
            #    self.centroids[m2].inc(d);
            #    continue;
            #if(c < maxCent):
            #    self.pushCent();
            #    c+=1;
            #    continue;
            #self.centroids[r[0]].inc(d);
        
        #return r2;

    #@numba.jit 
    def stepPassIV(self, batch, thres, maxCent):
        c = len(self.centroids);
        pT = 1e5;
        
        cn = 0;
        mask = np.empty([len(batch)], np.bool);
        mask[:] = True;
        #for self.step in batch:
        
        sdata = self.data[:, batch];
 
        while (pT > thres) & (cn < maxCent) & (sum(mask) > 0):
        
            #print("p4: ", cn, sum(mask), len(mask), sdata.shape);
            
            sdata = sdata[:, mask];
            
            mask = mask[mask];
            
            r2 = self.vstepf(self.centroids[cn], sdata[0,:], sdata[1, :]);
            
            xmask = r2 < thres;
            
            if(sum(xmask) > 0):
            
                self.centroids[cn].inc_Multi(sdata[:,xmask])
            
            mask[mask] = ~xmask;
            
            m2 = np.argmax(r2);
            v2 = r2[m2];
            
            if(v2<pT):
                pT = v2;
            
            cn += 1;
            if((pT > thres) & (cn ==c) & (sum(mask) > 0)):
                #self.step = m2;
                #self.pushCent();                
                self.centroids.append(liveCent(sdata[:,mask][:,0])) ##?
                c+=1;
    
    
class centKitII(centKit):

    #data;
    #velocity
    #centroids;
    #distances;/membership?
    #step    
    
    def pushCent(self):
        self.centroids.append(liveCentII([self.data[0][self.step], self.data[1][self.step]], [self.velocity[0][self.step], self.velocity[1][self.step]]));
    
    def __init__(self, xdata, xvel):
        self.data = self.initData(xdata);
        self.velocity = self.initData(xvel);
        self.step = 0;
        self.centroids = [];
        self.pushCent();
        self.member = np.empty([self.data.shape[1]], np.int16);
        self.member[:] = -1;
        
        self.vstepf = np.vectorize(liveCentII.dist_v); ##, excluded=['pointa', 'pointb']);

        
    def stepPassIV(self, batch, thres, maxCent):
        c = len(self.centroids);
        pT = 1e5;
        
        cn = 0;
        mask = np.empty([len(batch)], np.int16);
        mask[:] = -1;
        #for self.step in batch:
        xmask = mask == -1;
        
        sdata = self.data[:, batch];
        svel = self.velocity[:, batch];
        
 
        while (pT > thres) & (cn < maxCent) & (sum(xmask) > 0):
        
            #print("p4: ", cn, sum(mask), len(mask), sdata.shape);
            
            #sdata = sdata[:, mask];
            
            #xmask = mask == -1;
            
            r2 = self.vstepf(self.centroids[cn], sdata[0,xmask], sdata[1, xmask]);
            
            #zmask = r2 < thres;
            mask[xmask] = -1 + (1 + cn)*(r2<thres);
            
            if(sum(mask == cn) > 0):
            
                self.centroids[cn].inc_Multi(sdata[:,mask == cn])
            
            #mask[mask] = ~xmask;
            
            m2 = np.argmax(r2);
            v2 = r2[m2];
            
            if(v2<pT):
                pT = v2;
            
            cn += 1;
            
            xmask = mask == -1;
            
            if((pT > thres) & (cn ==c) & (sum(xmask) > 0)):
                #self.step = m2;
                #self.pushCent();                
                self.centroids.append(liveCentII(sdata[:,xmask][:,0], svel[:,xmask][:,0])) ##?
                c+=1;
        
        self.member[batch] = mask;
        
        
        
    def centVel(self):
        c = len(self.centroids);
        nM = self.member != -1;
        n = sum(nM);
        for cx in range(0, c):
            xm = self.member == cx;
            if(sum(xm > 0)):
                self.centroids[cx].velocity.inc_Multi(self.velocity[:, xm]);
        
        
class liveCentIII(liveCent):

    def __init__(self, cent, xrange):
        self.centre = cent;
        self.range = xrange;

        
class centKitIII():

    def __init__(self, xkit, xcent):
        #self.data = xkit.data;
        #self.velocity = self.initData(xvel);
        #self.step = 0;
        self.c_init = xcent
        self.centroids = [];
        for x in xkit.centroids:
            self.centroids.append(self.c_init(x.centre, x.range));
        #self.pushCent();
        self.member = np.empty([xkit.data.shape[1]], np.int16);
        self.member[:] = -1;
    
    def get_Member(self, data):
        n = len(data[0]);
        m = len(self.centroids);
        rmem = np.empty([n], np.int16);
        rmem[:] = -1;
        
        for i in range(0, n):            
            for j in range(0, m):
                if self.centroids[j].inrange([data[0,i], data[1, i]]):
                    rmem[i]=j;
                    break;
        return rmem;
                    
            
        #self.vstepf = np.vectorize(liveCentII.dist_v); ##, excluded=['pointa', 'pointb']);

class liveCentIV():

    def __init__(self, cent): #, xrange):
        self.centre = cent;
        #self.range = xrange;
        
    def dist_Floating(self, data, weight): 
                
        fval = np.mean(data - self.centre);  ##?
        
        return np.sum(np.abs(((data - fval) - self.centre) * weight))
    

    def dist_FloatEval(self, data, weight): 
                
        fval = np.sum((data - self.centre) * weight) / np.sum(weight);  ##?
        
        return np.sum(np.abs(((data - fval) - self.centre) * (0.+(weight==0)) ))


    def dist_FloatII(self, data, weight): 
                
        fval = np.sum((data - self.centre) * weight) / np.sum(weight); 
        
        return np.sum(np.abs(((data - fval) - self.centre) * weight ))


        
class centKitIV():
    
    def __init__(self, data, w):

        self.data = data;
        self.dshape = data.shape[1:];
        self.weight = w;
        
        self.centroids = np.empty(shape=[0], dtype=liveCentIV)
        self.distances = np.empty(shape=[self.data.shape[0]], dtype=float)
        print(self.dshape, self.weight.shape);
        

        
    def dist(self, dataind, centind):
        if (dataind<self.data.shape[0]) & (centind<self.centroids.shape[0]):    
            return np.sum(np.abs((self.data[dataind] - self.centroids[centind].centre) * self.weight))
        
        
    def distII(self, dataind, centind):
        if (dataind<self.data.shape[0]) & (centind<self.centroids.shape[0]):
            return self.centroids[centind].dist_Floating(self.data[dataind], self.weight); 
        #np.sum(np.abs((self.data[dataind] - self.centroids[centind].centre) * self.weight))
        
    def pushCent(self, dataind):
        self.centroids = np.append(self.centroids, liveCentIV(self.data[dataind]));    

    def distPass(self, centind):
        
        for i in range(0,self.data.shape[0]):        
            self.distances[i] = self.dist(i, centind);
            
    def distPassII(self, centind):
        for i in range(0,self.data.shape[0]): 
            self.distances[i] = self.distII(i, centind);
    
    def distPassIII(self, centind):
        for i in range(0,self.data.shape[0]): 
            self.distances[i] = self.centroids[centind].dist_FloatII(self.data[i], self.weight); ###
    
            
    def train(self, maxCen, maxThr): ##, d_fun = distPassII):
        
        c = len(self.centroids);
        ci=0;
        if (c==0):             
            self.pushCent(0);
            c += 1;            
            
        if (c>maxCen): c = maxCen;
            
        cont = True;
        cid = np.zeros(self.distances.shape, dtype=int);
        cid[:] = -1;
        #peakvals = np.zeros(self.distances.shape, dtype=float) + 1e7;
        unassign = -1;
        while(cont):
            
            self.distPassIII(ci); ##d_fun(self, ci);
            xid = np.nonzero(self.distances <= maxThr)[0];
            self.centroids[ci].centre = np.mean(self.data[xid], axis=0);
            cid[xid] = ci;
            ci += 1;
            if (ci >= maxCen):
                cont=False;
            if (ci >= c) & (cont):
                unassign = np.nonzero(cid==-1)[0];
                
                if len(unassign)>0:    
                    self.pushCent(unassign[0]);
                    c+= 1;
                    
                if len(unassign) == 0:
                    cont=False;


    
    def xLen(self, item):
        if type(item) == list:
            return len(item);
        if type(item) == numpy.ndarray:
            return item.shape[0];
        
    
    def distPassIIe(self, xval):
    
        n = len(self.centroids);
        
        bestVals = 1e7;
        bid = -1;
        cVal = 0;
        invC = 0;
        
        
        for i in range(0, n): 
            cVal = self.centroids[i].dist_FloatII(xval, self.weight);
            if cVal < bestVals:
                bestVals = cVal;
                bid = i;
                
        return self.centroids[bid].dist_FloatEval(xval, self.weight);
            
            
    
        
    def eval(self, val_set):
        
        n = val_set.shape[0]; ##self.xLen(val_set); 
        
        scores = np.zeros(n, dtype=float);
        
        for i in range(0,n):
            
            scores[i] = self.distPassIIe(val_set[i]);
            
        return scores;
        

        


##