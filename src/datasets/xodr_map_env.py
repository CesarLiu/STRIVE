# 
#   OpenDrive MapEnvironment based on BARK MapInterface to be used with STRIVE model (instead of NuScencesMapEnv)
#   fortiss, 2024
#

import bark

# for importing OpenDrive map
from bark.runtime.commons.xodr_parser import XodrParser

# for creating MapInterface
from bark.runtime.commons.parameters import ParameterServer
from bark.core.world import World
from bark.core.world.map import MapInterface
from bark.core.world.opendrive import XodrDrivingDirection, XodrLaneType

# for plotting BARK map and objects with MPViewer
from bark.runtime.viewer.matplotlib_viewer import MPViewer
import matplotlib.pyplot as plt
# from matplotlib.collections import LineCollection

# for using BARK geometry objects and functions
from bark.core.geometry import *

# for rasterization
import numpy as np

# for STRIVE MapEnv
import torch
import os
import datasets.nuscenes_utils as nutils

# analogous to NuScenesMapEnv in STRIVE
class XodrMapEnv():
    def __init__(self, map_data_path,
                        mname,
                        map_list =['DR_DEU_Merging_MT_v01_shifted', 'DR_DEU_Merging_MT_v01_centered', 'Crossing8Course', 'threeway_intersection', '4way_intersection'],
                        bounds=[-17.0, -38.5, 60.0, 38.5],
                        layers=['drivable_area', 'carpark_area', 'road_divider', 'lane_divider'],
                        L=256,
                        W=256,
                        device='cpu',
                        load_lanegraph=False,
                        lanegraph_res_meters=1.0,
                        lanegraph_eps=1e-6,
                        pix_per_m=4,
                        plot= True): 
        """ 
        Creates binary rasterized map image of OpenDrive map to be used with STRIVE model.
        The map image consists of four layers: drivable area, road dividers, lane dividers and carpark area. 
        The carpark area is set to 0 by default. (Carpark data is not available for Xodr map in BARK, since XodrLaneType 'PARKING' 
        is not implemented, see also commons.hpp.)
        
        Arguments:
            map_data_path (str): path to the dataset (e.g. /STRIVE/data/interaction) which contains the map directory
            map_list (list[str]): list of map names avaiable in map directory
            map_idx (int): index of map to be processed in list of maps (map_list)
            bounds (list[float]): [low_l, low_w, high_l, high_w] distances (in meters) around location to crop map observations
            layers (list[str]): name of the map layers to create from xodr map (based on NuScenes map layers)
            L (int): number of pixels along length of vehicle to render crop with
            W (int): number of pixels along width of vehicle to render crop with
            device (str): the device to store the rasterized maps on
            load_lanegraph (boolean): if true, loads the lane graph as well
            lanegraph_res_meters (float): resolution at which to discretize lane graph
            lanegraph_eps:
            pix_per_m (int): resolution to discretize map layers
            plot (boolean): True for generating plots of drivable area (for other map layers: uncomment lines below)
        """
        super(XodrMapEnv, self).__init__()

        self.data_path = map_data_path
        self.xodr_maps = {} # dictionary of xodr maps: {mname: xodr_map}
        self.map_list = map_list
        self.layer_names = layers
        self.bounds = bounds
        self.L = L
        self.W = W
        self.device = torch.device(device)
        self.num_layers = len(layers)

        # binarize the layers we need for the maps and cache for crop later
        print('Rasterizing Xodr maps...')
        m_per_pix = 1.0 / pix_per_m
        grid_size = m_per_pix
        self.xodr_raster = []
        self.xodr_dx = []
        max_H, max_W = -float('inf'), -float('inf')
        msize_list = []

        # choose map to be processed, creating interface for only one map
        self.map_name = mname
        midx =  map_list.index(mname)
        self.map_idx = midx # map identified by index map_idx in map_list
        
        # TODO: iterate over all maps in directory (create map_img for each and append to xodr_raster) using for loop
        # for midx, mname in enumerate(self.map_list): # indent until comment "# pad each map for efficient cropping" 

        # initialize BARK world and map interface
        params = ParameterServer()
        world = World(params)
        map_interface = MapInterface()
        world.SetMap(map_interface)

        # create MapInterface and get OpenDrive map
        map_file_name = os.path.join(map_data_path+str(mname)+'.xodr')
        self.map_file_name = map_file_name
        print(map_file_name)

        xodr_parser = XodrParser(map_file_name) 
        map_interface.SetOpenDriveMap(xodr_parser.map)
        xodr_map = map_interface.GetOpenDriveMap()
        # self.xodr_maps[mname] = xodr_map
        # for only one map
        self.xodr_maps = {mname: xodr_map}

        # load lane graphs
        if load_lanegraph:
            print('Loading lane graphs...')
            # TODO implement process_lanegraphs for Xodr map
            # self.lane_graphs = {map_name: nutils.process_lanegraph(nmap, lanegraph_res_meters, lanegraph_eps) for map_name,nmap in self.xodr_maps.items()}
        
        # get bounding box coordinates of full map
        map_bb = world.bounding_box # = (Point2d: x: -49.9756, y: -50.0556, Point2d: x: 49.9256, y: 50.1256)
        x_min, y_min = [0, 0] # for compatibility with NuScences and STRIVE functions
        # x_min, y_min = [map_bb[0].x(), map_bb[0].y()]
        x_max, y_max = [map_bb[1].x(), map_bb[1].y()]
        
        # add border to avoid leaving index range when getting map crops
        x_max += L/2
        y_max += W/2

        # initialize raster grid coordinates for full map
        x_grid = np.arange(x_min, x_max+grid_size, grid_size).T # want to include x_max in rasterized grid, but np.arange(start, stop, step) excludes stop
        y_grid = np.arange(y_min, y_max+grid_size, grid_size).T # include y_max
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid) # X_grid and Y_grid are numpy arrays
    
        # TODO grid size calculations analogous to STRIVE MapEnv (using msize to determine dx)
        # # map size in meters (H x W)
        # H = y_max - y_min # height (in meters)
        # W = x_max - x_min # width (in meters)
        # msize = np.array([H, W])

        #  msize (int) -- number of pixel (per axis)

        # for fixed grid_size (same for both direction): determine msize and dx from grid size
        cur_dx = (grid_size, grid_size)
        cur_msize = X_grid.shape
        cur_msize = tuple(cur_msize)
        self.xodr_dx.append(cur_dx)

        # re-calculate map size (in meters) from grid
        # H = x_grid[-1] - x_grid[0]
        # W = y_grid[-1] - y_grid[0]
        # msize = np.array([H, W])
        # msize (int) -- number of pixel (per axis)

        # get maximum map size (max_H, max_W) of all maps (for completion and compatibilty with STRIVE MapEnv also included for only one map)
        if cur_msize[0] > max_H:
            max_H = cur_msize[0]
        if cur_msize[1] > max_W:
            max_W = cur_msize[1]
        msize_list.append(cur_msize)

        # get binarized rasterization of full map
        map_layers = []
        
        # get all roads and their road ids
        roads_dict = xodr_map.GetRoads()
        road_ids = roads_dict.keys()
        road_ids = list(road_ids)

        # initialize driving direction
        driving_direction = XodrDrivingDirection.forward # XodrDrivingDirections: forwards, backwards, both
        
        # initizalize layer images
        img_size = (1,) + cur_msize # cur_msize = np.zeros(X_grid.shape)
        print('img_size: ', img_size)
        drivable_area_img = np.zeros(img_size) 
        road_divider_img = np.zeros(img_size)
        lane_divider_img = np.zeros(img_size)
        carpark_area_img = np.zeros(img_size)

        # iterate over all roads to extract information for each layer
        for road_id in road_ids:
            # get current road and its lane sections
            road = roads_dict[road_id] 
            lane_sections = road.lane_sections

            # generate road corridor of current road (includes only driving lanes)
            # TODO check if generating road corridor for one possible driving direction (e.g. forward) is enough
            map_interface.GenerateRoadCorridor([road_id], driving_direction)
            road_corridor = map_interface.GetRoadCorridor([road_id], driving_direction)

            # get road polygon of road corridor and rasterize it
            road_polygon = road_corridor.polygon 
            ix, iy, rasterized_road_polygon = self.RasterizePolygon(road_polygon, x_grid, y_grid, grid_size)

            # add rasterized polygon to drivable area
            drivable_area_img[0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += rasterized_road_polygon

            # iterate over lane sections
            for lane_section in lane_sections:
                #iterate over all lanes to get lane and road dividers
                for _, lane in lane_section.GetLanes().items():
                    lane_position = lane.lane_position
                    lane_left = lane_section.GetLaneByPosition(lane_position+1) # positive lane numbers
                    lane_right = lane_section.GetLaneByPosition(lane_position-1) # negative lane numbers
                    
                    # get road divider (center lane at position 0)
                    if lane_position == 0 and lane_left and lane_right:
                        lane_center = lane
                        center_line = lane_center.line
                        ix, iy, rasterized_center_line = self.RasterizeLine(center_line, x_grid, y_grid, grid_size)
                        road_divider_img[0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += rasterized_center_line  
                        continue
                    
                    # only interested in dividers between drivable lanes
                    if lane.lane_type != XodrLaneType.driving:
                        continue
                        
                    # get lane divider (btw lanes going in same direction)
                    divider_line = Line2d()
                    if lane_position > 0 and lane_left and lane_left.lane_type==XodrLaneType.driving:
                        left_boundary = lane_left.line
                        divider_line = left_boundary
                
                    elif lane_position < 0 and lane_right and lane_right.lane_type==XodrLaneType.driving:
                        right_boundary = lane.line
                        divider_line = right_boundary
                    
                    ix, iy, rasterized_divider_line = self.RasterizeLine(divider_line, x_grid, y_grid, grid_size)
                    lane_divider_img[0, iy[0]:iy[-1]+1, ix[0]:ix[-1]+1] += rasterized_divider_line  

        # limit values in map layers to range [0,1]
        drivable_area_img = np.clip(drivable_area_img, 0, 1)
        road_divider_img = np.clip(road_divider_img, 0, 1)
        lane_divider_img = np.clip(lane_divider_img, 0, 1)
        
        # append all layers to map layers
        print('drivable_area_img: ', np.array(drivable_area_img).shape)
        map_layers.append(drivable_area_img)
        map_layers.append(road_divider_img)
        map_layers.append(lane_divider_img)
        map_layers.append(carpark_area_img)
        # print('map_layers: ', np.array(map_layers).shape)
        
        if plot:
            # plot map layers
            self.PlotRasterMap(x_grid, y_grid, drivable_area_img[0], grid_size, 'drivable_area')
            # self.PlotRasterMap(x_grid, y_grid, road_divider_img[0], grid_size, 'road_divider') 
            # self.PlotRasterMap(x_grid, y_grid, lane_divider_img[0], grid_size, 'lane_divider')
        
        # create single image
        map_img = np.concatenate(map_layers, axis=0)
        print('map_img: ', map_img.shape)
  
        # pad each map to same size (max_w, max_H) 
        pad_right = max_W - cur_msize[1]
        pad_bottom = max_H - cur_msize[0]
        pad = torch.nn.ZeroPad2d((0, pad_right, 0, pad_bottom))
        padded_map_img = pad(torch.from_numpy(map_img).unsqueeze(0))[0]

        self.xodr_raster.append(padded_map_img)

        # pad each map to max for efficient cropping
        self.xodr_raster = torch.stack(self.xodr_raster, dim=0).to(device)
        self.xodr_dx = torch.from_numpy(np.stack(self.xodr_dx, axis=0)).to(device)
        
        print('xodr_raster: ', self.xodr_raster.shape)
        
        # needed for compatibility with STRIVE functions
        # TODO check where these variables are called in orginal STRIVE functions and fix names
        self.nusc_raster = self.xodr_raster
        self.nusc_dx = self.xodr_dx

    def RasterizePolygon(self, polygon, x_grid, y_grid, grid_size=None):
        """ Creates rasterized image of polygon based on given raster grid and grid size.

        Arguments:
            polygon (Polygon2d) -- BARK geometry polygon to be rasterized
            x_grid (np.array) -- vector of x coordinates in global map grid
            y_grid (np.array) -- vector of y coordinates in global map grid
            grid_size (float) -- delta between grid coordinates on x- and y-axis

        Returns:
            ix (np.array) -- vector of x indices of polygon in global map grid
            iy (np.array) -- vector of y indices of polygon in global map grid
            rasterized_polygon (np.array) -- binary mask of polygon
        """
        # get bounding box coordinates of polygon
        polygon_bb = polygon.bounding_box
        
        # get min and max coordinates from bounding box
        x_min, x_max = [polygon_bb[0].x(), polygon_bb[1].x()]
        y_min, y_max = [polygon_bb[0].y(), polygon_bb[1].y()]

        # calculate grid_size if necessary
        if grid_size is None:
            grid_size = x_grid[1] - x_grid[0] # = y_grid[1] - y_grid[0]

        # extract grid vectors for polygon from global grid of drivable area
        x_grid_poly  = x_grid[(x_grid > x_min-grid_size) & (x_grid < x_max+grid_size)]
        y_grid_poly = y_grid[(y_grid > y_min-grid_size) & (y_grid < y_max+grid_size)]

        # get coordinate grids from coordinate vectors
        X_grid_poly, Y_grid_poly = np.meshgrid(x_grid_poly, y_grid_poly)

        # get all grid points in raster grid of polygon (stacked in one array of pairs)
        grid_points = np.vstack([X_grid_poly.ravel(), Y_grid_poly.ravel()])
        grid_points = grid_points.T #np.transpose(grid_points)

        # get indices of polygon in full map image
        ix = np.where((x_grid > x_min-grid_size) & (x_grid < x_max+grid_size))[0]
        iy = np.where((y_grid > y_min-grid_size) & (y_grid < y_max+grid_size))[0]

        # initialize rasterized polygon
        rasterized_polygon = np.zeros(X_grid_poly.shape) # = Y_grid_poly.shape

        # initialize indices for iterating over grid points
        idx = 0
        idy = 0
        
        # iterate over grid points and check whether point us in road polygon using Within() function from BARK
        rasterized_polygon = np.zeros(X_grid_poly.shape)
        for point in grid_points:
            # get x and y coordinates of current point
            x,y = point

            # create BARK geometry point and check if point lies within polygon (excluding points on boundary line)
            pt = Point2d(x,y)
            if Within(pt, polygon):
                # set value to 1 if point is within polygon
                rasterized_polygon[idy, idx] = 1

            #increment indices
            idx+=1
            if idx > (X_grid_poly.shape[1]-1):
                idx = 0
                idy +=1

        return ix, iy, rasterized_polygon


    def RasterizeLine(self, line, x_grid, y_grid, grid_size=None):
        """ Creates rasterized image of line based on given raster grid and grid size.

        Arguments:
            line (Line2d) -- BARK geometry line to be rasterized
            x_grid (np.array) -- vector of x coordinates in global map grid
            y_grid (np.array) -- vector of y coordinates in global map grid
            grid_size (float) -- delta between grid coordinates on x- and y-axis

        Returns:
            ix (np.array) -- vector of x indices of polygon in global map grid
            iy (np.array) -- vector of y indices of polygon in global map grid
            rasterized_line (np.array) -- binary mask of line
        """
        # get bounding box coordinates of polygon
        line_bb = line.bounding_box
        
        # get min and max coordinates from bounding box
        x_min, x_max = [line_bb[0].x(), line_bb[1].x()]
        y_min, y_max = [line_bb[0].y(), line_bb[1].y()]

        # calculate grid_size if necessary
        if grid_size is None:
            grid_size = x_grid[1] - x_grid[0] # = y_grid[1] - y_grid[0]

        # extract grid vectors for polygon from global grid of drivable area
        x_grid_line  = x_grid[(x_grid > x_min-grid_size) & (x_grid < x_max+grid_size)]
        y_grid_line = y_grid[(y_grid > y_min-grid_size) & (y_grid < y_max+grid_size)]

        # get coordinate grids from coordinate vectors
        X_grid_line, Y_grid_line = np.meshgrid(x_grid_line, y_grid_line)

        # get indices of line in full map image
        ix = np.where((x_grid > x_min-grid_size) & (x_grid < x_max+grid_size))[0]
        iy = np.where((y_grid > y_min-grid_size) & (y_grid < y_max+grid_size))[0]

        # get all line points
        line_points = line.ToArray()

        # initialize rasterized polygon
        rasterized_line = np.zeros(X_grid_line.shape) # = Y_grid_line.shape
        for i in range(len(line_points) - 1):
            x1, y1 = line_points[i]
            x2, y2 = line_points[i + 1]
            x0 = x_grid_line[0]
            y0 = y_grid_line[0]
            idx1 = np.floor((x1 - x0) / grid_size).astype(int)
            idy1 = np.floor((y1 - y0) / grid_size).astype(int)
            idx2 = np.floor((x2 - x0) / grid_size).astype(int)
            idy2 = np.floor((y2 - y0) / grid_size).astype(int)
            rasterized_line[idy1, idx1] = 1
            rasterized_line[idy2, idx2] = 1
            points = np.array([idx1, idy1, idx2, idy2]).reshape(2, 2)
            for x, y in self.BresenhamLine(*points[0], *points[1]):
                rasterized_line[y, x] = 1
        
        return ix, iy, rasterized_line


    def BresenhamLine(self, x0, y0, x1, y1):
        """ Rasterizes line between two points (x0, y0) and (x1, y1) using Bresenham line algorithm.
            Note that coordinates are int values because they represent indices.

        Arguments:
            (x0, y0) (np.array) -- coordinates of first point on line
            (x1, y1) (np.array) -- coordinates of second point on line

        Returns:
            (x, y) (np.array) -- sequences of coordinates of points on rasterized line
        """
        # calculate derivations in x and y direction
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        steep = dy > dx # steep slope, larger than 1

        if steep: # reverse input coordinates so that dx > dy (slope between 0 and 1)
            steep = True
            x0, y0 = y0, x0
            x1, y1 = y1, x1
            
        if x0 > x1: # reverse order of input points
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        # update derivations
        dx = x1 - x0
        dy = abs(y1 - y0)

        # determine step size 
        sy = 1 if y0 < y1 else -1 # = ystep
        # sx = 1 if x0 < x1 else -1 # = xstep # included in for-loop
        
        # initialize max error and coordinates
        error = dx / 2.0
        y = y0 
        # x = x0 # included in for-loop
        
        for x in range(x0, x1 + 1): # step = sx = 1 for x0 < x1
            # 'yield' (similar to 'return') -- generates sequence of points (instead of one specific value)
            yield (y, x) if steep else (x, y)
            error -= dy
            # x += sx # performed by for-loop
            if error < 0:
                y += sy
                error += dx


    def PlotRasterMap(self, x_grid, y_grid, raster_map, grid_size, layer_name):
        """ Plots binary image of one rasterized map layer.

        Arguments:
            x_grid (np.array) -- vector of grid coordinates along x-axis
            y_grid (np.array) -- vector of grid coordinates along y-axis
            raster_img (np.array) -- rasterized map image as binary array containing 0's and 1's
            grid_size (float) -- delta between grid coordinates on x- and y-axis
            layer_name (string) -- name of layer (drivable area, road divider, lane divider, carpark area)

        Returns:
            plot of rasterized map as binary image (mask)
        """
        # plot rasterized map image
        plt.imshow(raster_map, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], 
                origin='lower', cmap='binary')
        
        # add axis labels and title
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('rasterized map of '+layer_name+' with grid size '+str(grid_size))
        plt.show()

    def get_map_crop(self, scene_graph, map_idx,
                    bounds=None,
                    L=None,
                    W=None):
        '''
        Render local crop for whole batch of agents represented as a scene graph.
        Assumes .pos is UNNORMALIZED in true scale map coordinate frame.

        :param scene_graph: batched scene graph with .pos size (N x 4) or (N x NS x 4) (x,y,hx,hy) in .lw (N x 2) (x,y)
                            will render each crop in the frame of .pos. The .batch attrib must be idx
                            for which agent is in which batch.
        :param map_idx: the map index of each batch in the scene graph (B,)
        :params bounds, L, W: overrides bounds, L, W set in constructor

        :returns map_crop: N x C x H x W
        '''
        device = scene_graph.pos.device
        B = map_idx.size(0)
        NA = scene_graph.pos.size(0)

        bounds = self.bounds if bounds is None else bounds
        L = self.L if L is None else L
        W = self.W if W is None else W

        # map index for each agent in the whole scene graph. 
        mapixes = map_idx[scene_graph.batch]
        pos_in = scene_graph.pos
        if len(scene_graph.pos.size()) == 3:
            NS = scene_graph.pos.size(1)
            pos_in = pos_in.reshape(NA*NS, -1)
            mapixes = mapixes.unsqueeze(1).expand(NA, NS).reshape(-1)

        # render by indexing into pre-rasterized binary maps
        map_obs = nutils.get_map_obs(self.xodr_raster, self.xodr_dx, pos_in,
                                     mapixes, bounds, L=L, W=W).to(device)
       
        return map_obs
    
    # copy paste from NuScenesMapEnv
    def objs2crop(self, center, obj_center, obj_lw, map_idx, bounds=None, L=None, W=None):
        '''
        converts given objects N x 4 to the crop frame defined by the given center (x,y,hx,hy)
        '''
        bounds = self.bounds if bounds is None else bounds
        L = self.L if L is None else L
        W = self.W if W is None else W
        local_objs = nutils.objects2frame(obj_center.cpu().numpy()[np.newaxis, :, :],
                                          center.cpu().numpy())[0]
        # [low_l, low_w, high_l, high_w]
        local_objs[:, 0] -= bounds[0]
        local_objs[:, 1] -= bounds[1]

        # convert to pix space
        pix2m_L = L / float(bounds[2] - bounds[0])
        pix2m_W = W / float(bounds[3] - bounds[1])
        local_objs[:, 0] *= pix2m_L
        local_objs[:, 1] *= pix2m_W
        pix_objl = obj_lw[:, 0]*pix2m_L
        pix_objw = obj_lw[:, 1]*pix2m_W
        pix_objlw = torch.stack([pix_objl, pix_objw], dim=1)
        local_objs = torch.from_numpy(local_objs)

        return local_objs, pix_objlw

