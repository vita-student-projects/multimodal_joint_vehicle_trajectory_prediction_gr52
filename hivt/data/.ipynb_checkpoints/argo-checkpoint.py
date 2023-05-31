"""
Module implementing MyData dataset, as the dataloader class.
"""
from typing import Dict

import torch
from omegaconf import DictConfig

from trajpred._internal.stats import Split
from trajpred.data_handling.base_traj_dataset import BaseTrajPredDataset
from trajpred.data_handling.dataset_adapters.argo_adapter import ArgoAdapter

from trajpred.baselines.hivt.data.HiVT_adapter import HiVTAdapter

class ArgoDataset(BaseTrajPredDataset):
    '''
    '''

    def __init__(self, config: DictConfig, data_split: str = "train") -> None:
        # avg len_lane_centerlines 104.381  avg len_lane_cl 9.99970682101842

        if data_split == "train":
            split_option = config.data.argo.train_split
        elif data_split == "val" or data_split == "eval":
            split_option = config.data.argo.val_split
        elif data_split == "test":
            split_option = config.data.argo.test_split
        else:
            raise ValueError("split must be one of [train, val, test]")

        config.model.remove_static_agnts = False
        super(ArgoDataset, self).__init__(config, data_split=data_split)
        self.dataset_adapter = ArgoAdapter(split_option, config, data_split=data_split)
        self.model_adapter = HiVTAdapter(config, data_split=data_split)
        print('argo line 35')
        #breakpoint()
        print(f"Initialized {self.__class__.__name__} with {len(self)} samples.")

    def __len__(self) -> int:
        return len(self.dataset_adapter)

    @staticmethod
    def manipulate_data(data: Dict, gt: Dict) -> Dict:
        return data, gt