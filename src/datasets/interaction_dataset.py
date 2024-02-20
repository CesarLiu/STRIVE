import os, itertools

import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data as Graph
from torch_geometric.data import DataLoader as GraphDataLoader

from utils.interaction_dataset.interaction_dataset_scenario_generation import InteractionDatasetScenarioGeneration

import sys, os
cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))
import datasets.nuscenes_utils as nutils
from datasets.utils import MeanStdNormalizer, normalize_scene_graph, read_adv_scenes, NUSC_NORM_STATS
from datasets.xodr_map_env import XodrMapEnv

import bark
from bark.runtime.commons.parameters import ParameterServer
from bark.examples.paths import Data

class InteractionDataset(Dataset):
    def __init__(self, bark_scenarios,  
                map_env,
                categories=['car', 'truck'],
                npast=4,
                nfuture=12,
                noise_std=0.0,
                scenario_path=None
              ):
        '''
        Creates dataset by taking BARK scenarios (generated from INTERACTION data) and map environment as input and 
        storing selected information about ego and other agents and the map in dataset.

        Arguments:
          - bark_scenarios: list of BARK scenario objects (generated from INTERACTION dataset)
          - map_env : map environment to base map indices on
          - categories : which types of agents to return from
                          ['car', 'bicycle', 'pedestrian']
          - npast : the number of input (past) steps
          - nfuture : the number of output (future) steps
          - noise_std: standard dev of gaussian noise to add to state vector
          - scenario_path: path to STRIVE scenarios (json) to load in as the dataset
        '''
        super(InteractionDataset, self).__init__()
        self.scenario_list = bark_scenarios
        self.use_bark = self.scenario_list is not None
        self.map_env = map_env
        self.map_list = self.map_env.map_list
        self.dt = 0.1 # 100 ms = 10 Hz
        self.noise_std = noise_std
        self.npast = npast
        self.nfuture= nfuture
        self.seq_len = npast + nfuture

        # setting attributes to False when not using NuScenes data
        self.use_nusc = False
        self.use_challenge_splits = False
        self.require_full_past = False

        self.scenario_path = scenario_path
        if self.scenario_path is None:
            assert self.use_bark

        # keep categories from NuScenes for compatibiltiy
        all_cats = ['car', 'truck', 'bus', 'motorcycle', 'trailer', 'cyclist', 'pedestrian', 'emergency', 'construction']
        all_cat2key = {
            'car' : ['vehicle.car'],
            'truck' : ['vehicle.truck'],
            'bus' : ['vehicle.bus'],
            'motorcycle' : ['vehicle.motorcycle'],
            'trailer' : ['vehicle.trailer'],
            'cyclist' : ['vehicle.bicycle'],
            'pedestrian' : ['human.pedestrian'],
            'emergency' : ['vehicle.emergency'],
            'construction' : ['vehicle.construction']
        }

        self.categories = categories
        self.key2cat = {}
        for cat in self.categories:
            if cat not in all_cats:
                print('Unrecognized category %s!' % (cat))
                exit()
            for k in all_cat2key[cat]:
                self.key2cat[k] = cat

        iden = torch.eye(len(self.categories), dtype=torch.int)
        self.cat2vec = {self.categories[cat_idx] : iden[cat_idx] for cat_idx in range(len(self.categories))} # needed for get_item()
        self.vec2cat = {tuple(iden[cat_idx].tolist()) : self.categories[cat_idx]  for cat_idx in range(len(self.categories))} # needed to complie_scenarios()

        # tally number of frames in each class
        self.data = {}
        self.seq_map = []
        self.scene2map = {}
        print('Loading in scenario data...')
        if self.use_bark and self.scenario_list is not None:
            print('Using BARK scenarios...')
            scenario_data, scenario_seq_map, scenario_scene2map = self.compile_bark_scenarios(self.scenario_list)
        else:
            print('Using scenario files...')
            scenario_data, scenario_seq_map, scenario_scene2map = self.compile_scenarios(self.scenario_path)
        
        self.data = {**self.data, **scenario_data}
        self.seq_map = self.seq_map + scenario_seq_map
        self.scene2map = {**self.scene2map, **scenario_scene2map}
        print('Num adversarial subseq: %d' % (len(scenario_seq_map)))

        self.data_len = len(self.seq_map)
          
        print('Num scenes: %d' % (len(self.data)))
        print('Num subseq: %d' % (self.data_len))

        # build normalization info objects
        # state normalizer. states of (x, y, hx, hy, s, hdot)
        ninfo = NUSC_NORM_STATS[tuple(sorted(self.categories))]
        norm_mean = [ninfo['lscale'][0], ninfo['lscale'][0], ninfo['h'][0], ninfo['h'][0], ninfo['s'][0], ninfo['hdot'][0]]
        norm_std = [ninfo['lscale'][1], ninfo['lscale'][1], ninfo['h'][1], ninfo['h'][1], ninfo['s'][1], ninfo['hdot'][1]]
        self.normalizer = MeanStdNormalizer(torch.Tensor(norm_mean),
                                            torch.Tensor(norm_std))
        # vehicle attribute normalizer of (l, w)
        att_norm_mean = [ninfo['l'][0], ninfo['w'][0]]
        att_norm_std = [ninfo['l'][1], ninfo['w'][1]]
        self.veh_att_normalizer = MeanStdNormalizer(torch.Tensor(att_norm_mean),
                                                  torch.Tensor(att_norm_std))
        self.norm_info = ninfo


    def compile_bark_scenarios(self, bark_scenario_list):
        """
        Extracts information about ego and other agent (including trajectories) and map from list of scenarios and 
        returns them in dictionary.

        Arguments:
            bark_scenario_list : list of BARK scenarios
        
        Returns:
            scene2info : dictionary with ego and other agent info for each scene
            seq_map : mapping of scene name to start index (default: start_idx = 0) 
            scene2map : mapping of scene name to map name and index
        """
        scene2info = {}
        scene2map = {}
        seq_map = []

        # scenario list only contains one scenario (num_scenarios=1) for interaction scenario (replay)
        scene = bark_scenario_list[0]
        world_state = scene.GetWorldState()
        # TODO create a list containing more than one BARK scenario and iterate over scenarios using for-loop:
        # for scene in bark_scenario_list: # indent until end of function (except for line "return...")

        # TODO customize scene name
        sname = 'interaction-example'
        scene2info[sname] = {}

        all_tracks = []
        lw = []

        # get trajectories of scenario for each timestep and agent
        for (agent_id, agent) in world_state.agents.items():
            vehicle_shape = agent.shape # BARK polygon(Polgyon2d) 
            bb = vehicle_shape.bounding_box
            vehicle_length = bb[1].x() - bb[0].x() # = wb + 2*r with wb: wheelbase, r: collision radius (from BARK)
            vehicle_width  = bb[1].y() - bb[0].y() # = 2*r
            vehicle_lw = (vehicle_length, vehicle_width) 
            lw.append(vehicle_lw)

            behavior = agent.behavior_model
            agent_traj = behavior.static_trajectory
            # agent_traj -- list of BARK agent states: y(t, i) = [x(t), y(t), h(t), v(t)]
            
            agent_track = [] 
            for state in agent_traj: 
                t, x, y, h, v = state[0], state[1], state[2], state[3], state[4] 
                # with  (x, y) -- 2D position
                #       h = theta (int, rad) -- heading angle
                #       v -- speed
                # cur_state = (x, y, h, v)
                # transform: h -- heading angle --> (hx, hy) -- heading unit vector
                hx = np.cos(h) # h must be in rad!
                hy = np.sin(h)
                # hvec = np.array(hx, hy)
                cur_state = (x, y, hx, hy, v) # (x, y, hvec, v)
                agent_track.append(cur_state)
            # agent tracks must have same starting times and length
            all_tracks.append(agent_track)
                                
        lw = np.array(lw)
        all_tracks = np.array(all_tracks) # shape: (NA, nsteps, 5) -- state vector with 5 elements
        NA = all_tracks.shape[0] # NA: number of agents
        t = np.arange(all_tracks.shape[1])*self.dt
        k = ['car' for _ in range(NA)] # by default assume all cars

        all_tracks_h = all_tracks[:,:,2]
        hdots = [nutils.heading_change_rate(all_tracks_h[aidx], t)[0:] for aidx in range(NA)] # adix -- agent index
        # hdot -- heading change rate (yaw rate)
        all_tracks = np.dstack((all_tracks, hdots))
        # all_tracks -- list of agent states, including velocity and heading change rate [x(t), y(t), hx(t), hy(t), v(t), hdot(t)]

        # ego
        # TODO this would be a good point to set ego vehicle manually, based on ID
        # get ego_id from config (-> scenario_generation._ego_track_id)
        ego_id = 8 # for TrackFile track_05_short.csv
        # get index of ego agent (from ego_id)
        ego_idx = 4 # for TrackFile track_05_short.csv
        ego_traj = all_tracks[ego_idx]
        ego_lw = lw[ego_idx]
        ego_is_vis = np.ones((ego_traj.shape[0]), dtype=int)

        ego_info = {
            'traj' : ego_traj,
            'lw' :  ego_lw,
            'is_vis' : ego_is_vis,
            'k' : 'ego',
        }
        scene2info[sname]['ego'] = ego_info
        
        # all others
        for aidx in [idx for idx in range(NA) if idx != ego_idx]:
            cur_traj = all_tracks[aidx]
            is_vis = np.logical_not(np.isnan(np.sum(cur_traj, axis=1))).astype(int)
            info = {
                'traj' : cur_traj,
                'lw' :  lw[aidx],
                'is_vis' : is_vis,
                'k' : k[aidx]
            }
            scene2info[sname]['agt%03d' % (aidx)] = info
        
        # update data map
        mname = self.map_env.map_name
        scene2map[sname] = (mname, self.map_list.index(mname)) #(scene['map'], self.map_list.index(scene['map']))
        T = ego_traj.shape[0]
        # scene_seq = [(sname, start_idx) for start_idx in range(0, T - self.seq_len, 1)]
        scene_seq = [(sname, 0)] # because T = self.seq_len when using strive scenario generated with default values.
        seq_map.extend(scene_seq)
        
        return scene2info, seq_map, scene2map

    def compile_scenarios(self, scenario_path):
        adv_scenes = read_adv_scenes(scenario_path)
        scene2info = {}
        scene2map = {}
        seq_map = []
        for scene in adv_scenes:
            sname = scene['name']
            scene2info[sname] = {}

            lw = scene['veh_att'].numpy()
            NA = lw.shape[0]
            k = ['car' for _ in range(NA)] # by default assume all cars
            if 'sem' in scene:
                k = [self.vec2cat[tuple(sem)] for sem in scene['sem'].numpy().tolist()]

            past = scene['scene_past'].numpy()
            fut = scene['scene_fut'].numpy()

            attack_t = scene['attack_t']
            
            # need to compute velocities for future (already have for past)
            fut_traj = np.concatenate([past[:, -1:, :4], fut], axis=1)
            t = np.arange(fut_traj.shape[1])*self.dt
            vels = [np.linalg.norm(nutils.velocity(fut_traj[aidx, :, :2], t)[1:], axis=1) for aidx in range(NA)]
            fut_h = np.arctan2(fut_traj[:, :, 3], fut_traj[:, :, 2])
            hdots = [nutils.heading_change_rate(fut_h[aidx], t)[1:] for aidx in range(NA)]
            
            # ego
            ego_fut = np.concatenate([fut[0], vels[0][:, np.newaxis], hdots[0][:, np.newaxis]], axis=1)
            ego_traj = np.concatenate([past[0], ego_fut], axis=0)
            ego_lw = lw[0]
            ego_is_vis = np.ones((ego_traj.shape[0]), dtype=int)
            ego_info = {
                'traj' : ego_traj,
                'lw' :  ego_lw,
                'is_vis' : ego_is_vis,
                'k' : 'ego',
            }
            scene2info[sname]['ego'] = ego_info

            # all others
            for aidx in range(1, NA):
                cur_fut = np.concatenate([fut[aidx], vels[aidx][:, np.newaxis], hdots[aidx][:, np.newaxis]], axis=1)
                cur_traj = np.concatenate([past[aidx], cur_fut], axis=0)
                is_vis = np.logical_not(np.isnan(np.sum(cur_traj, axis=1))).astype(int)
                info = {
                    'traj' : cur_traj,
                    'lw' :  lw[aidx],
                    'is_vis' : is_vis,
                    'k' : k[aidx]
                }
                scene2info[sname]['agt%03d' % (aidx)] = info
            
            # update data map
            scene2map[sname] = (scene['map'], self.map_list.index(scene['map']))
            T = ego_traj.shape[0]
            # scene_seq = [(sname, start_idx) for start_idx in range(0, T - self.seq_len, 1)]
            scene_seq = [(sname, 0)] # because T = self.seq_len when using strive scenario generated with default values.
            seq_map.extend(scene_seq)
        
        return scene2info, seq_map, scene2map 

    def __len__(self):
        return self.data_len
  
    def __getitem__(self, idx):
        idx_info = self.seq_map[idx]
        inst_tok = None
        scene_name, sidx = idx_info
        eidx = sidx + self.seq_len
        midx = sidx + self.npast
        _, map_idx = self.scene2map[scene_name]

        # NOTE only keep an agent in the sequence if it has an annotation
        #       at last frame of the past.
        #       This is not perfect since past/future-only agents will certainly affect traffic

        # always put ego at node 0
        ego_data = self.data[scene_name]['ego']
        past = [ego_data['traj'][sidx:midx, :]]
        future = [ego_data['traj'][midx:eidx, :]]
        sem = [self.cat2vec['car']] # one-hot vec
        lw = [ego_data['lw']]
        past_vis = [ego_data['is_vis'][sidx:midx]]
        fut_vis = [ego_data['is_vis'][midx:eidx]]

        for agent in self.data[scene_name]:
            if agent == 'ego':
                continue
            if self.use_challenge_splits and agent == inst_tok:
                continue
            agent_data = self.data[scene_name][agent]
            if np.isnan(agent_data['traj'][midx-1]).astype(np.int).sum() > 0:
                continue
            if self.require_full_past and np.isnan(agent_data['traj'][:midx]).sum() > 0:
                # has some nan in past
                continue

            # have a valid agent, add info
            # may be nan at many frames, this must be dealt with in model
            past.append(agent_data['traj'][sidx:midx, :])
            future.append(agent_data['traj'][midx:eidx, :])
            sem.append(self.cat2vec[agent_data['k']])
            lw.append(agent_data['lw'])
            past_vis.append(agent_data['is_vis'][sidx:midx])
            fut_vis.append(agent_data['is_vis'][midx:eidx])

        past = torch.Tensor(np.stack(past, axis=0))
        future = torch.Tensor(np.stack(future, axis=0))
        sem = torch.Tensor(np.stack(sem, axis=0))
        lw = torch.Tensor(np.stack(lw, axis=0))
        past_vis = torch.Tensor(np.stack(past_vis, axis=0))
        fut_vis = torch.Tensor(np.stack(fut_vis, axis=0))

        # normalize
        past_gt = self.normalizer.normalize(past) # gt past (no noise)
        past = self.normalizer.normalize(past)
        future_gt = self.normalizer.normalize(future) # gt future (used to compute err/loss)
        future = self.normalizer.normalize(future) # observed future (input to net)
        lw = self.veh_att_normalizer.normalize(lw)

        # add noise if desired
        if self.noise_std > 0:
            past += torch.randn_like(past)*self.noise_std
            future += torch.randn_like(future)*self.noise_std
            # make sure heading is still a unit vector
            past[:, :, 2:4] = past[:, :, 2:4] / torch.norm(past[:, :, 2:4], dim=-1, keepdim=True)
            future[:, :, 2:4] = future[:, :, 2:4] / torch.norm(future[:, :, 2:4], dim=-1, keepdim=True)
            # make sure position is still positive
            past[:, :, :2] = torch.clamp(past[:, :, :2], min=0.0)
            future[:, :, :2] = torch.clamp(future[:, :, :2], min=0.0)
            # also for vehicle attributes
            lw += torch.randn_like(lw)*self.noise_std

        #  then build fully-connected scene graph
        NA = past.size(0)
        edge_index = None
        if NA > 1:
            node_list = range(NA)
            edge_index = list(itertools.product(node_list, node_list))
            edge_index_list = [(i, j) for i, j in edge_index if i != j]
            edge_index = torch.Tensor(edge_index_list).T.to(torch.long).contiguous()
        else:
            edge_index = torch.Tensor([[],[]]).long()

        graph_prop_dict = {
            'x' : torch.empty((NA,)),
            'pos' : torch.empty((NA,)),
            'edge_index' : edge_index,
            'past' : past,
            'past_gt' : past_gt,
            'future' : future,
            'future_gt' : future_gt,
            'sem' : sem,
            'lw' : lw,
            'past_vis' : past_vis,
            'future_vis' : fut_vis,
        }
        scene_graph = Graph(**graph_prop_dict)

        return scene_graph, map_idx
    
    def get_state_normalizer(self):
        return self.normalizer
    
    def get_att_normalizer(self):
        return self.veh_att_normalizer