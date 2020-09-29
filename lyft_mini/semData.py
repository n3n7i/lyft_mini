

import Nelsa
import numpy as np # linear algebra

class mapII(Nelsa.mapI):
    
    def tce(self, elem):
        return elem.traffic_control_element

    def pdelta(self, elem):
        return {'px': self.cm_Norm(elem.points_x_deltas_cm), 'py': self.cm_Norm(elem.points_y_deltas_cm), 'pz': self.cm_Norm(elem.points_z_deltas_cm)};
    
    def ldelta(self, elem):
        return {'px': self.cm_Norm(elem.vertex_deltas_x_cm), 'py': self.cm_Norm(elem.vertex_deltas_y_cm), 'pz': self.cm_Norm(elem.vertex_deltas_z_cm)};
    
    
    pass;

class mapData:
    
    def scan_Labels(self, xstr):
        return [self.map.label_Mask(i, xstr) for i in range(0, self.maplen)];
    
    def collect_Labels(self, bvec):
        return [self.map[int(x)] for x in np.nonzero(bvec)[0]];
    
    
    def delta_Offsets(self):
        for x in self.traffics:    
            x.delta_Offset = self.map.gps_to_World(x.Centre);
    
        
    def __init__(self, xmap):
        self.lanes = [];
        self.traffics = [];
        self.junctions = [];
        self.map = xmap; ##self.map.Nelsa_?
        self.maplen = len(xmap);
        
        self.setlanes(self.collect_Labels(self.scan_Labels("lane")));
        self.settraffic(self.collect_Labels(self.scan_Labels("traffic_control_element")));
        self.setjunction(self.collect_Labels(self.scan_Labels("junction")));
        
        for x in range(0, len(self.lanes)):
            self.lanes[x] = self.lane(self.lanes[x]);

        for x in range(0, len(self.traffics)):
            self.traffics[x] = self.traffic(self.traffics[x]);
            
        self.delta_Offsets();


    class common:        
        #def __init__(self, xob):
        #    self.obj = xob;

        def __init__(self, xob):
            self.obj = xob;
            self.Id = self.get_Id();
            self.Centre = self.get_Centroid();
        
        def get_Id(self):
            return self.obj.id.id;
        
        def get_Centroid(self):
            #print(type(self));
            #if(type(self) == mapData.traffic): print("trafficlight");
            #if(type(self) == mapData.lane): print("lane");
            bound = self.obj.bounding_box;
            lat = (bound.south_west.lat_e7 + bound.north_east.lat_e7) / 2e7
            long = (bound.south_west.lng_e7 + bound.north_east.lng_e7) / 2e7
            return [lat, long];##?
            #pass;
            
        def get_Delta(self):
            #print(type(self), type(self.obj))
            if(type(self) == mapData.traffic):
            
                return [mapII.pdelta(self, self.obj.element.traffic_control_element)];
            
            if(type(self) == mapData.lane):
                
                lane1 = mapII.ldelta(self, self.obj.element.lane.left_boundary);
                lane2 = mapII.ldelta(self, self.obj.element.lane.right_boundary);
                
                #print(self.obj == mII[self.obj.get_Id()])
                #return mII.pdelta(mII[self.obj.get_Id()]); ##
                return [lane1, lane2];
            #pass;
            
        def cm_Norm(self, seq):    ## to metres absolute
            return np.cumsum(np.asarray(seq) / 100)

                

        
        #pass;
        
    class node(common): pass;
#        def __init__(self, xob):
#            self.obj = xob;
            ## road segments
            ## junction
            
    class segment(common): pass;
#        def __init__(self, xob):
#            self.obj = xob;
            ## start node / end node
            ## lanes

    class junction(common): pass;
#        def __init__(self, xob):
#            self.obj = xob;
            ## road net nodes
            ## traffic
            ## lanes

    class lane(common):
        def __init__(self, xob):
            self.obj = xob;
            self.Id = self.get_Id();
            self.Centre = self.get_Centroid();
            self.Delta = self.get_Delta();
    
    class traffic(common):
        def __init__(self, xob):
            self.obj = xob;
            self.Id = self.get_Id();
            self.Centre = self.get_Centroid();
            self.Delta = self.get_Delta();
            self.delta_Offset = np.array([0., 0.]);

    
    def setlanes(self, xl):
        self.lanes.extend(xl);
        
    def settraffic(self, xt):
        self.traffics.extend(xt);
    
    def setjunction(self, xt):
        self.junctions.extend(xt);
    
#