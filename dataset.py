import numpy as np
from ome_zarr.reader import Reader as ZarrReader
from ome_zarr.io import parse_url

import os
import math
import json

from matplotlib import pyplot as plt

class TomoTiles:
    @staticmethod
    def read_zarr(zarr_path: str) -> np.ndarray:
        """Finds array with highest resolution and returns it as a np array"""

        zarr_url = parse_url(zarr_path)
        reader = ZarrReader(zarr_url)
        image_node = list(reader())[0]
        dask_data = image_node.data

        # Find element with highest resolution
        dask_array_resolutions = [math.prod(d.shape) for d in dask_data]
        idx_high_res = np.argmax(dask_array_resolutions)

        # Convert to np array
        # np_array = dask_data[idx_high_res].compute()
        np_array = dask_data[0].compute()
        return np_array

    def __init__(self, tomo_dir):
        """Assumes tomo_dir contains one .zarr file and n>0 .ndjson annotation files."""
        # Get zarr file
        zarr_files = [f for f in os.listdir(tomo_dir) if f.endswith(".zarr")]
        if len(zarr_files) != 1:
            raise ValueError(f"Expected exactly one .zarr file, found {len(zarr_files)}")
        self.zarr_file = os.path.join(tomo_dir, zarr_files[0])
        
        # Get annotations
        self.annotation_files = [
            os.path.join(tomo_dir, f) 
            for f in os.listdir(tomo_dir) 
            if f.endswith(".ndjson")
        ]

        # Get tomogram metadata
        tomo_info_path = os.path.join(tomo_dir, 'tomo-metadata.json')
        with open(tomo_info_path, 'r') as f:
            self.tomo_info = json.load(f)
        
        # Save tomogram shape data
        self.tomo_shape = np.array([
            self.tomo_info['size_x'], 
            self.tomo_info['size_y'], 
            self.tomo_info['size_z']
        ])
    
    def voxels_to_angstroms(self, vox):
        return vox * self.tomo_info['voxel_spacing'] 
    
    def angstroms_to_voxels(self, ang):
        return ang / self.tomo_info['voxel_spacing']

    def tile_with_points(self, tiles_dir: str='tiles', tile_size_angstroms=(256, 512, 512)):
        """Tile up the tomogram, retaining and transforming any point annotations pertaining to this tomogram."""
        # Convert tile size from tuple or list to np.ndarray if necessary
        if not isinstance(tile_size_angstroms, np.ndarray):
            tile_size_angstroms = np.array(tile_size_angstroms)

        point_locs = []
        for ann_file in self.annotation_files:
            with open(ann_file, "r") as f:
                anns = [json.loads(line) for line in f]
            for ann in anns:
                if ann['type'] != 'point':
                    continue
                point_locs.append([ # Reverse order to match C-ordering for numpy
                    ann['location']['z'],
                    ann['location']['y'],
                    ann['location']['x'],
                ])
        
        # Full tomogram array
        tomo_array = self.read_zarr(self.zarr_file)

        # Divide up array
        tomo_shape_angstroms = self.voxels_to_angstroms(self.tomo_shape)

        # Calculate how many tiles we can fit in each dimension
        # tile_num_per_dim = 




if __name__ == '__main__':
    data_path = '/home/mward19/nobackup/autodelete/fm-data-2'
    tiler = TomoTiles(os.path.join(data_path, 'tomo-10456'))
    tiler.tile_with_points()

