import os
import sys
import pyprojroot
sys.path.append(pyprojroot.here())
from torch.utils.data import Dataset
import numpy as np
import torch
from pathlib import Path
from ..preprocessing.subtile_esd_hw02 import Subtile, TileMetadata
from typing import List, Dict
from copy import deepcopy
class DSE(Dataset):
    """
    Custom dataset for the IEEE GRSS 2021 ESD dataset.

    args:
        root_dir: str | os.PathLike
            Location of the processed subtiles
        selected_bands: Dict[str, List[str]] | None
            Dictionary mapping satellite type to list of bands to select
        transform: callable, optional
            Object that applies augmentations to a sample of the data
    attributes:
        root_dir: str | os.PathLike
            Location of the processed subtiles
        tiles: List[Path]
            List of paths to the subtiles
        transform: callable
            Object that applies augmentations to the sample of the data

    """
    def __init__(self, root_dir: str | os.PathLike, selected_bands: Dict[str, List[str]] | None=None, transform=None):
        self.root_dir = Path(root_dir)
        self.selected_bands = selected_bands
        self.transform = transform
        self.tiles = []

        for parent_tile in [ f.path for f in os.scandir(self.root_dir) if f.is_dir() ]:
            self.tiles.extend([os.path.join(parent_tile, "subtiles", subtile) for subtile in os.listdir(os.path.join(parent_tile, "subtiles"))])
        

    def __len__(self):
        """
            Returns number of tiles in the dataset

            Output: int
                length: number of tiles in the dataset
        """
        return len(self.tiles)

    def __aggregate_time(self, img):
        """
            Aggregates time dimension in order to 
            feed it to the machine learning model.

            This function needs to be changed in the
            final project to better suit your needs.

            For homework 2, you will simply stack the time bands
            such that the output is shaped (time*bands, width, height),
            i.e., all the time bands are treated as a new band.

            Input:
                img: np.ndarray
                    (time, bands, width, height) array
            Output:
                new_img: np.ndarray
                    (time*bands, width, height) array
        """
        return img.reshape(-1, img.shape[2], img.shape[3])

    
    def __select_indices(self, bands: List[str], selected_bands: List[str]):
        """
            Selects the indices of the bands used.

            Input:
                bands: List[str]
                    list of bands in the order that they are stacked in the
                    corresponding satellite stack
                selected_bands: List[str]
                    list of bands that have been selected

            Output:
                bands_indices: List[int]
                    index location of selected bands
        """
        indexlist = []
        for band in selected_bands:
            indexlist.append(bands.index(band))
        return indexlist
    
    def __select_bands(self, subtile):
        """
            Aggregates time dimension in order to
            feed it to the machine learning model.

            This function needs to be changed in the
            final project to better suit your needs.

            For homework 2, you will simply stack the time bands
            such that the output is shaped (time*bands, width, height),
            i.e., all the time bands are treated as a new band.

            Input:
                subtile: Subtile object
                    (time, bands, width, height) array
            Output:
                selected_satellite_stack: Dict[str, np.ndarray]
                    satellite--> np.ndarray with shape (time, bands, width, height) array

                new_metadata: TileMetadata
                    Updated metadata with only the satellites and bands that were picked
        """
        new_metadata = deepcopy(subtile.tile_metadata)
        if self.selected_bands is not None:
            selected_satellite_stack = {}
            new_metadata.satellites = {}
            for key in self.selected_bands:
                satellite_bands = subtile.tile_metadata.satellites[key].bands
                selected_bands = self.selected_bands[key]
                indices = self.__select_indices(satellite_bands, selected_bands)
                new_metadata.satellites[key] = subtile.tile_metadata.satellites[key]
                subtile.tile_metadata.satellites[key].bands = self.selected_bands[key]
                # for i in indices:
                #   selected_satellite_stack[key][:, i, :, :] = subtile.satellite_stack[key][:, i, :, :]
                selected_satellite_stack[key] = subtile.satellite_stack[key][:, indices, :, :] # dimensions [t, bands, h, w]
        else:
            selected_satellite_stack = subtile.satellite_stack

        return selected_satellite_stack, new_metadata

    def __getitem__(
            self,
            idx: int
            ) -> tuple[np.ndarray, np.ndarray, TileMetadata]:
        """
            Loads subtile at index idx, then
                - selects bands
                - aggregates times
                - stacks satellites
                - performs self.transform
            
            Input:
                idx: int
                    index of subtile with respect to self.tiles
            
            Output:
                X: np.ndarray | torch.Tensor
                    input data to ML model, of shape (time*bands, width, height)
                y: np.ndarray | torch.Tensor
                    ground truth, of shape (1, width, height)
                tile_metadata:
                    corresponding tile metadata
        """
        # load the subtiles using the Subtile class in
        # src/preprocessing/subtile_esd_hw02.py
        subtile_path = self.tiles[idx]
        #OR THIS ONE IDK subtile = load(subtile_path)
        subtile = Subtile(None, None)
        subtile.load(os.path.join(subtile_path)) 

        # call the __select_bands function to select the bands and satellites
        selected_sat_stack, new_metadata = self.__select_bands(subtile)
        
        X = None
        for key in selected_sat_stack.keys():
            if key != "gt":
                temp = selected_sat_stack[key]
                curr = self.__aggregate_time(temp)
                if type(X) == np.ndarray:
                    X = np.concatenate((X, curr))
                else:
                    X = curr

        # Concatenate the time and bands

        # Adjust the y ground truth to be the same shape as the X data by
        # removing the time dimension
        y = subtile.satellite_stack["gt"][0, 0, :, :].reshape(1, subtile.tile_metadata.subtile_size, subtile.tile_metadata.subtile_size)

        # all timestamps are treated and stacked as bands

        # if there is a transform, apply it to both X and y
        if self.transform:
            vectDict = {"X" : X, "y" : y}
            vectDict = self.transform(vectDict)
            X = vectDict ["X"]
            y = vectDict["y"]
        
        return (X, y-1, new_metadata)