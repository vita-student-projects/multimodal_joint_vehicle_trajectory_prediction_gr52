"""
Preprocess the data and store it. This file has base dataloader and some useful functions to write a preprocessing code.
If a preprocessing is not required, just re-store the data

Note: place your code in the blank region in preprocess_dataset.
"""

import os
import pickle

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from trajpred.config import Config
from trajpred.entrypoint import Entrypoint
from trajpred.data_handling.dataset_adapters.dataset_utils import to_long, to_int16, to_numpy, from_numpy
from utils import TemporalData

os.umask(0)

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')  # FIXME Check if that works fine


@hydra.main(config_path="../../../../config/hydra", config_name="config.yaml")
def main(cfg: DictConfig):
    print("Starting preprocessing -------------")

    # Import all settings for experiment.
    cfg.device = "cpu"
    entrypoint = Entrypoint(cfg)
    config = entrypoint._get_configs(train=False)
    config.hydra["preprocessed"] = False  # we use raw data to generate preprocess data
    # print(f"{cfg.data.waymo.split_preprocess_num} out of {cfg.data.num_splits_preprocess} splits")

    save_dir = config.hydra.data[config.hydra.data_features.dataset_name.lower()]['preprocess_train']
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    train(entrypoint, config)
    breakpoint()
    val(entrypoint, config)
    # test(entrypoint, config)
    print("Finished preprocessing  -------------")

def train(entrypoint: Entrypoint, config: Config):
    # Data loader for training set
    dataset = next(iter(entrypoint.create_train_datasets(config).values()))
    preprocess_dataset(dataset, config, 'train')


def test(entrypoint: Entrypoint, config: Config):
    # Data loader for test set
    dataset = next(iter(entrypoint.create_eval_datasets(config).values()))
    preprocess_dataset(dataset, config, 'test')


def val(entrypoint: Entrypoint, config: Config):
    # Data loader for val set
    dataset = next(iter(entrypoint.create_val_datasets(config).values()))
    preprocess_dataset(dataset, config, 'val')


def preprocess_dataset(dataset: Entrypoint, config: Config, mode: str = 'train'):  # FIXME Use split enum
    # Data loader for training set
    mode = "test" if mode == "eval" else mode
    data_loader = DataLoader(
        dataset,
        batch_size=config.hydra.data_loader.batch_size,
        num_workers=config.hydra.data_loader.load_num_workers,
        shuffle=False,
        collate_fn=collate_fn, 
        pin_memory=False,
        drop_last=False,
    )

    store = [None for _ in range(len(data_loader.dataset))]
    for i, data in enumerate(tqdm(data_loader, total=len(data_loader))):
        if data is None:
            continue
        '''
        ********************************
        your preprocessing code can go here.
        the final preprocessed data should be called "store"
        ********************************
        '''
        store[i] = TemporalData(**data[0])


    ## saving preprocessed files.
    dataset_name = config.hydra.data_features.dataset_name
    if dataset_name == "Argo":  # FIXME Have consistent naming convention
        dataset_name = "argo"
    elif dataset_name == "Argo2":
        dataset_name = "av2"
    save = config.hydra.data[dataset_name.lower()][f"preprocess_{mode}"]
    

    # Save pkl files individually
    samples_per_split = 1
    if config.hydra.data.waymo.root_version == "mini":
        samples_per_split = 41
    print("Number of files: ", len(store), " not valid: ", len([s for s in store if s is None]))
    for i in range(len(store)):
        if store[i] is None:
            continue
        total_idx = samples_per_split * config.hydra.data.split_preprocess_num + i
        save_file_str = f"{save[:-2]}_{total_idx:02}_{str(store[i].seq_id)}.p"
        with open(save_file_str, 'wb') as f:
            pickle.dump(store[i], f, protocol=pickle.HIGHEST_PROTOCOL)
                                                 
                                                          
    # Save config file                                                     
    dataset_name = config.hydra.data_features.dataset_name
    save_dir = config.hydra.data[dataset_name.lower()]['preprocess_train']
    OmegaConf.save(config.hydra,
                   f"{os.path.dirname(save_dir)}/preprocess_config_{config.hydra.data.split_preprocess_num}_out_of_{config.hydra.data.num_splits_preprocess}.yaml")
    print(
        f"Config file saved to {os.path.dirname(save_dir)}/preprocess_config_{config.hydra.data.split_preprocess_num}_out_of_{config.hydra.data.num_splits_preprocess}.yaml")


class PreprocessDataset():
    def __init__(self, split):
        self.split = split

    def __getitem__(self, idx):
        data = self.split[idx]
        if data is None:
            return None

        return data

    def __len__(self):
        return len(self.split)


def collate_fn(batch):
    #batch = from_numpy(batch)
    #return_batch = dict()
    ## Batching by use a list for non-fixed size
    #i = 0
    #while batch[i] is None:
    #    i += 1
    #for key in batch[i].keys():
    #    # return_batch[key] = [x[key] for x in batch]
    #    return_batch[key] = [x[key] for x in batch if x is not None]
    return batch


if __name__ == "__main__":
    main()
