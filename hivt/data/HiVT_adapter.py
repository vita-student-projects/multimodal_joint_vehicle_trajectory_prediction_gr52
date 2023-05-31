from typing import Dict, Union, Tuple, List
from trajpred.data_handling.base_model_adapter import BaseModelAdapter
import argparse

import h5py
import os
from itertools import permutations
from itertools import product
import time

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader
from tqdm import tqdm

from typing import Callable, Dict, List, Optional, Tuple, Union
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
from trajpred import tptensor
from trajpred.tptensor import TPTensor
from trajpred.baselines.hivt.utils import TemporalData
#TODO: check for path root, raw file name, processed file names


#TODO: delete if runs smoothly without
#also delete get_obj_feats and just keep the num_timesteps etc outside the func
#Since the theta is computed the same yaw/theta as in HiVT 
#Otherwise we do the work twice




class HiVTAdapter(BaseModelAdapter):
    '''
    This class is used to adapt loaded data to the template model
    '''
    def __init__(self,
                 root: str,
                 data_split: str,
                 #transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        self.data_split = data_split
        self._local_radius = local_radius
        #self._url = f'https://s3.amazonaws.com/argoai-argoverse/forecasting_{split}_v1.1.tar.gz'
        if data_split == 'sample':
            self._directory = 'forecasting_sample'
        elif data_split == 'train':
            self._directory = 'train'
        elif data_split == 'val':
            self._directory = 'val'
        elif data_split == 'test':
            self._directory = 'test_obs'
        else:
            raise ValueError(data_split + ' is not valid')
        self.root = root
        #self._raw_file_names = os.listdir(self.raw_dir)
        #self._processed_file_names = [os.path.splitext(f)[0] + '.pt' for f in self.raw_file_names]
        #self._processed_paths = [os.path.join(self.processed_dir, f) for f in self._processed_file_names]
        super(HiVTAdapter, self).__init__(root) #transform=transform in the parenthesis
        
    def get_obj_feats(self, data: Dict) -> Dict:
        if hasattr(self, 'len_lane_cl'):
            pass
        else:
            self.len_lane_cl = 0
            self.len_lane_centerlines = 0
            self.counter_pnts = 0
            self.counter_lane = 0

        feat_cfg = self.config.data_features
        max_pnts_per_lane = self.config.model.max_pnts_per_lane
        num_timesteps = feat_cfg.num_future_ts + feat_cfg.num_past_ts
        obs_timesteps = feat_cfg.num_past_ts
        pred_timesteps = feat_cfg.num_future_ts

        max_num_roads = self.config.model.max_num_roads  # manually found
        max_num_agents = self.config.model.max_num_agents  # manually found

        ego_traj = np.concatenate((np.array(data['trajs'][0]), np.ones( (len(data['trajs'][0]),1))), axis=1)


        others_traj = data['trajs'][1:]
    
        am = ArgoverseMap()
        split = self.data_split
        radius = 50
        df = data
    
        # filter out actors that are unseen during the historical time steps

        
        timestamps = data['steps']
        #print(len(data['trajs']),'\n \n' ,data['trajs'][0].shape)
        historical_timestamps = timestamps[:][: obs_timesteps]
        actor_ids = list(data['trajs'])
        num_nodes = len(data['trajs'])
    
        #av_df = data[data['OBJECT_TYPE'] == 'AV'].iloc
        #av_index = actor_ids.index(av_df[0]['TRACK_ID'])
        #av_index = actor_ids.index(data['trajs'][0])
        av_index = feat_cfg.agent_idx
        #av_index = data['trajs'][0]
        #agent_df = data[data['OBJECT_TYPE'] == 'AGENT'].iloc
        agent_index = list(range(1,len(data['trajs']))) 
        #actor_ids.index(data['idx'][1]) 
        #let's try with the first agent :) hoping it will always be the first the agent and not the second or third (=! other)
        
        city = data['city'][0]
    
        # make the scene centered at AV
        origin = torch.tensor([data['trajs'][0][obs_timesteps-1,0], data['trajs'][0][obs_timesteps-1,1]], dtype=torch.float)
        av_heading_vector = origin - torch.tensor([data['trajs'][0][obs_timesteps-2,0],
                                                   data['trajs'][0][obs_timesteps-2,1]], dtype=torch.float)
        #theta for the ego vehicle
        theta = torch.atan2(av_heading_vector[1], av_heading_vector[0])
        rotate_mat = torch.tensor([[torch.cos(theta), -torch.sin(theta)],
                                   [torch.sin(theta), torch.cos(theta)]])
    
        # initialization
        x = torch.zeros(num_nodes, num_timesteps, 2, dtype=torch.float)
        edge_index = torch.LongTensor(list(permutations(range(num_nodes), 2))).t().contiguous()
        padding_mask = torch.ones(num_nodes, num_timesteps, dtype=torch.bool)
        bos_mask = torch.zeros(num_nodes, obs_timesteps, dtype=torch.bool)
        rotate_angles = torch.zeros(num_nodes, dtype=torch.float)
        
        for i in range(len(data['trajs'])):
            
            #node_idx = actor_ids.index(actor_id)
            node_idx = i
            node_steps = data['steps'][i]
            #node_steps = [timestamps.index(timestamp) for timestamp in actor_df['steps']]
            padding_mask[node_idx, node_steps] = False
            if padding_mask[node_idx, obs_timesteps-1]:  # make no predictions for actors that are unseen at the current time step
                padding_mask[node_idx, obs_timesteps:] = True
            xy = torch.from_numpy(np.stack([data['trajs'][i][:,0],
                                            data['trajs'][i][:,1]], axis=-1)).float()
            x[node_idx, node_steps] = torch.matmul(xy - origin, rotate_mat)
            node_historical_steps = list(filter(lambda node_step: node_step < obs_timesteps, node_steps))
            if len(node_historical_steps) > 1:  # calculate the heading of the actor (approximately)
                heading_vector = x[node_idx, node_historical_steps[-1]] - x[node_idx, node_historical_steps[-2]]
                rotate_angles[node_idx] = torch.atan2(heading_vector[1], heading_vector[0])
            else:  # make no predictions for the actor if the number of valid time steps is less than 2
                padding_mask[node_idx, obs_timesteps:] = True
    
        # bos_mask is True if time step t is valid and time step t-1 is invalid
        bos_mask[:, 0] = ~padding_mask[:, 0]
        bos_mask[:, 1: obs_timesteps] = padding_mask[:, : obs_timesteps-1] & ~padding_mask[:, 1: obs_timesteps]
    
        positions = x.clone()
        x[:, obs_timesteps:] = torch.where((padding_mask[:, obs_timesteps-1].unsqueeze(-1) | padding_mask[:, obs_timesteps:]).unsqueeze(-1),
                                torch.zeros(num_nodes, pred_timesteps, 2),
                                x[:, obs_timesteps:] - x[:, obs_timesteps-1].unsqueeze(-2))
        x[:, 1: obs_timesteps] = torch.where((padding_mask[:, : obs_timesteps-1] | padding_mask[:, 1: obs_timesteps]).unsqueeze(-1),
                                  torch.zeros(num_nodes, obs_timesteps-1, 2),
                                  x[:, 1: obs_timesteps] - x[:, : obs_timesteps-1])
        x[:, 0] = torch.zeros(num_nodes, 2)
        #print(x.shape, '\n\n\t', x[0])
    
        # get lane features at the current time step
                                                             
        #df_19 = df[df['TIMESTAMP'] == timestamps[19]]
        #df_19 = data[data['steps']==timestamps[obs_timesteps-1]]                                                      
        node_inds_19 = list(range(len(data['trajs']))) 
        #maybe need to change df_19['trajs'][:,0] to df_19['trajs'][:,:,0], but shouldnt be necessary since we only have one
        node_positions_19 = torch.from_numpy(np.stack([x[:,obs_timesteps-1,0], x[:,obs_timesteps-1,1]], axis=-1)).float()
                                                             
        (lane_vectors, is_intersections, turn_directions, traffic_controls, lane_actor_index,
        lane_actor_vectors) = get_lane_features(am, data, node_inds_19, node_positions_19, origin, rotate_mat, city, radius)
        del data['map_pnts'] # delete raw map to reduce redundant data
        y = None if split == 'test' else x[:, obs_timesteps:]
        raw_path = "/Users/annevaleriepreto/Documents/EPFL/MA-2/Deep_learning/argodataset/train/data"
        #seq_id = os.path.splitext(os.listdir(raw_path))[0]
        seq_id = data["idx"]
        print(seq_id)
        #breakpoint()
        #print('in Preprocess HiVT adapter, just needs to save and run')
        del data
        #print("Preprocess begins")
        
        #data={}
        #
        #data['x']= x[:, : obs_timesteps]  # [N, 20, 2]
        #data['positions']= positions  # [N, 50, 2]
        #data['edge_index']= edge_index  # [2, N x N - 1]
        #data['y']= y  # [N, 30, 2]
        #data['num_nodes']= num_nodes
        #data['padding_mask']= padding_mask  # [N, 50]
        #data['bos_mask']= bos_mask  # [N, 20]
        #data['rotate_angles']= rotate_angles  # [N]
        #data['lane_vectors']= lane_vectors  # [L, 2]
        #data['is_intersections']= is_intersections  # [L]
        #data['turn_directions']= turn_directions  # [L]
        #data['traffic_controls']= traffic_controls  # [L]
        #data['lane_actor_index']= lane_actor_index  # [2, E_{A-L}]
        #data['lane_actor_vectors']= lane_actor_vectors  # [E_{A-L}, 2]
        #data['seq_id']= int(seq_id)
        #data['av_index']= av_index
        #data['agent_index']= agent_index
        #data['city']= city
        #data['origin']= origin.unsqueeze(0)
        #data['theta']= theta
        return {
            'x': x[:, : 20],  # [N, 20, 2]
            'positions': positions,  # [N, 50, 2]
            'edge_index': edge_index,  # [2, N x N - 1]
            'y': y,  # [N, 30, 2]
            'num_nodes': num_nodes,
            'padding_mask': padding_mask,  # [N, 50]
            'bos_mask': bos_mask,  # [N, 20]
            'rotate_angles': rotate_angles,  # [N]
            'lane_vectors': lane_vectors,  # [L, 2]
            'is_intersections': is_intersections,  # [L]
            'turn_directions': turn_directions,  # [L]
            'traffic_controls': traffic_controls,  # [L]
            'lane_actor_index': lane_actor_index,  # [2, E_{A-L}]
            'lane_actor_vectors': lane_actor_vectors,  # [E_{A-L}, 2]
            'seq_id': int(seq_id),
            'av_index': av_index,
            'agent_index': agent_index,
            'city': city,
            'origin': origin.unsqueeze(0),
            'theta': theta,
        }
                                                             
def get_lane_features(am: ArgoverseMap, data,
                      node_inds: List[int],
                      node_positions: torch.Tensor,
                      origin: torch.Tensor,
                      rotate_mat: torch.Tensor,
                      city: str,
                      radius: float,) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
                                          torch.Tensor]:
    lane_positions, lane_vectors, traffic_controls, turn_directions, is_intersections = [], [], [], [], []
    lane_ids = tuple(data['map_pnts']['lane_ids'])
    
    for lane_id in lane_ids :
        turn_direction = data["map_pnts"]["lanes"][lane_id].turn_direction
        lane_centerline = data["map_pnts"]["lanes"][lane_id].centerline[:, :2] #slicing as in original HiVT
        is_intersection = data["map_pnts"]["lanes"][lane_id].is_intersection
        traffic_control = data["map_pnts"]["lanes"][lane_id].has_traffic_control
        lane_positions.append(torch.from_numpy(lane_centerline[:-1])) #convert to torch, needed for torch.cat
        lane_vectors.append(torch.from_numpy(lane_centerline[1:] - lane_centerline[:-1]))
        count = len(lane_centerline) - 1
        
        if turn_direction == 'NONE':
            turn_direction = 0
        elif turn_direction == 'LEFT':
            turn_direction = 1
        elif turn_direction == 'RIGHT':
            turn_direction = 2
        else:
            raise ValueError('turn direction is not valid')
                                                         
        turn_directions.append(turn_direction * torch.ones(count, dtype=torch.uint8))
        traffic_controls.append(traffic_control * torch.ones(count, dtype=torch.uint8))
                                                       
    #for node_position in node_positions:
    #    lane_ids.update(am.get_lane_ids_in_xy_bbox(node_position[0], node_position[1], city, radius))
    node_positions = torch.matmul(node_positions - origin, rotate_mat).float()
    
    #torch.cat need a tuple of torch Tensors --> lane_positions should be that 
    lane_positions = tuple(lane_positions)    
    lane_positions = torch.cat(lane_positions, dim=0)
    lane_vectors = tuple(lane_vectors)
    lane_vectors = torch.cat(lane_vectors, dim=0)
                                                         
    lane_actor_index = torch.LongTensor(list(product(torch.arange(lane_vectors.size(0)), node_inds))).t().contiguous()
    lane_actor_vectors = \
        lane_positions.repeat_interleave(len(node_inds), dim=0) - node_positions.repeat(lane_vectors.size(0), 1)
    mask = torch.norm(lane_actor_vectors, p=2, dim=-1) < radius
    lane_actor_index = lane_actor_index[:, mask]
    lane_actor_vectors = lane_actor_vectors[mask]
    

    return lane_vectors, is_intersections, turn_direction, traffic_controls, lane_actor_index, lane_actor_vectors
                                                             
#Probably corresponds to our collate 
class ArgoverseV1DataModule(LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_transform: Optional[Callable] = None,
                 val_transform: Optional[Callable] = None,
                 local_radius: float = 50) -> None:
        super(ArgoverseV1DataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.local_radius = local_radius

    def prepare_data(self) -> None:
        HiVTAdapter(self.root, 'train', self.train_transform, self.local_radius)
        HiVTAdapter(self.root, 'val', self.val_transform, self.local_radius)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = HiVTAdapter(self.root, 'train', self.train_transform, self.local_radius)
        self.val_dataset = HiVTAdapter(self.root, 'val', self.val_transform, self.local_radius)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers,
                          pin_memory=self.pin_memory, persistent_workers=self.persistent_workers)