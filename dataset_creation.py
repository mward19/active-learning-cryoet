import numpy as np
from ome_zarr.reader import Reader as ZarrReader
from ome_zarr.io import parse_url
from tqdm.auto import tqdm

import os
import math
import json
from itertools import product
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import argparse

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
        self.tomo_dir = tomo_dir

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
        self.tomo_shape = np.array([ # Reverse because x, y, z reflect fortran ordering but numpy uses C-ordering
            self.tomo_info['size_z'], 
            self.tomo_info['size_y'], 
            self.tomo_info['size_x']
        ])

        self.tile_num_per_dim = None
        self.original_to_tiles = None
        self.tile_to_original = None

        self.tiles_dir = None
    
    def voxels_to_angstroms(self, vox):
        return vox * self.tomo_info['voxel_spacing'] 
    
    def angstroms_to_voxels(self, ang):
        return ang / self.tomo_info['voxel_spacing']


    def get_tiling_functions(self, tile_size_angstroms: np.ndarray, min_overlap_angstroms: np.ndarray):
        # Divide up tomogram array

        tomo_shape_angstroms = self.voxels_to_angstroms(self.tomo_shape)

        # Calculate how many tiles we can fit in each dimension 
        # (cover whole tomogram with minimal overlap between tiles)
        step_size_angstroms = tile_size_angstroms - min_overlap_angstroms
        self.tile_num_per_dim = np.ceil(tomo_shape_angstroms / step_size_angstroms).astype(int)
        # Get tile division points
        self.division_points = [
            np.linspace(
                0, 
                self.tomo_shape[i], 
                self.tile_num_per_dim[i],
                endpoint=False
            ) 
            for i in range(3)
        ]

        # Function to convert original loc to tile loc (in voxels)
        def original_to_tiles(original_loc):
            original_loc = np.array(original_loc)
            tiles = []

            for d in range(3):
                # Find all tiles along this dimension that contain the voxel
                # Tile covers from division_point to division_point + tile_size
                candidates = []
                for i in range(len(self.division_points[d]) - 1):
                    start = self.division_points[d][i]
                    end = self.division_points[d][i + 1]
                    if start <= original_loc[d] < end:
                        candidates.append(i)
                    # Also include previous tile if voxel falls in the overlap
                    elif i > 0 and original_loc[d] < start and original_loc[d] >= start - self.min_overlap_voxels[d]:
                        candidates.append(i - 1)
                if not candidates:
                    # If voxel is exactly at the end boundary
                    candidates.append(len(self.division_points[d]) - 2)
                if d == 0:
                    tile_indices = [[i] for i in candidates]
                else:
                    # Expand combinations across dimensions
                    tile_indices = [prev + [i] for prev in tile_indices for i in candidates]

            # Convert tile indices to tile_loc
            for idx in tile_indices:
                tile_loc = np.array([
                    original_loc[d] - self.division_points[d][idx[d]]
                    for d in range(3)
                ])
                tiles.append((np.array(idx), tile_loc))

            return tiles

        
        def tile_to_original(tile_idx, tile_loc=np.array([0, 0, 0])):
            tile_idx = np.array(tile_idx)
            tile_loc = np.array(tile_loc)
            return np.array([
                self.division_points[d][tile_idx[d]] + tile_loc[d]
                for d in range(3)
            ])

        # Save functions as class attributes
        self.original_to_tiles = original_to_tiles
        self.tile_to_original = tile_to_original

        return self.original_to_tiles, self.tile_to_original
    
    def tile_tomogram_points(
            self, 
            tile_size_angstroms: np.ndarray=np.array([1280, 2560, 2560]), 
            min_overlap_angstroms: np.ndarray=np.array([100, 100, 100]), # To ensure a target is not split between tiles 
            tiles_dir: str='tiles', 
            overwrite=False
        ):
        """Tile up the tomogram, retaining and transforming any point annotations pertaining to this tomogram."""
        # if self.original_to_tiles is None or self.tile_to_original is None or self.tile_num_per_dim is None:
        self.get_tiling_functions(tile_size_angstroms, min_overlap_angstroms)
        tile_size_voxels = self.angstroms_to_voxels(tile_size_angstroms)

        # Full tomogram array
        tomo_array = self.read_zarr(self.zarr_file)

        if np.any(tomo_array.shape != self.tomo_shape):
            raise ValueError(
                f"Tomogram array shape in tomo-metadata was {self.tomo_shape} "
                f"but did not match the actual tomogram array shape {tomo_array.shape}"
            )
        
        tiles_dir_full = os.path.join(self.tomo_dir, tiles_dir)
        os.makedirs(tiles_dir_full, exist_ok=True)
        self.tiles_dir = tiles_dir_full

        # Figure out which tile each point belongs to and where in that tile
        point_locs_by_tile = defaultdict(list)
        for ann_file in self.annotation_files:
            with open(ann_file, "r") as f:
                anns = [json.loads(line) for line in f]
            for ann in anns:
                if ann['type'] != 'point':
                    continue
                point_original = np.array([
                    ann['location']['z'],
                    ann['location']['y'],
                    ann['location']['x'],
                ])
                for tile_idx, tile_loc in self.original_to_tiles(point_original):
                    key = tuple(int(i) for i in tile_idx)
                    value = tuple(int(i) for i in tile_loc)
                    point_locs_by_tile[key].append(value)
            
        def write_points_to_json(json_path, points: list):
            if len(points) == 0:
                return
            with open(json_path, 'w') as f:
                json.dump(points, f, indent=4)

        # Start separating and saving tiles
        all_tile_indices = list(product(
            range(int(self.tile_num_per_dim[0])), 
            range(int(self.tile_num_per_dim[1])), 
            range(int(self.tile_num_per_dim[2]))
        ))

        def save_tile(indices, pbar):
            i, j, k = indices

            tile_top_corner = self.tile_to_original(np.array([i, j, k]))
            tile_bottom_corner = tile_top_corner + tile_size_voxels

            # Extract slice
            tc = tile_top_corner.astype(int)
            bc = tile_bottom_corner.astype(int)
            tile_array = tomo_array[tc[0]:bc[0], tc[1]:bc[1], tc[2]:bc[2]]

            # Save it as np.ndarray
            tile_name = f'tile-{i}-{j}-{k}'
            this_tile_dir = os.path.join(tiles_dir_full, tile_name)
            os.makedirs(this_tile_dir, exist_ok=True)
            tile_npy_path = os.path.join(this_tile_dir, 'tile.npy')
            if overwrite or not os.path.isfile(tile_npy_path):
                np.save(tile_npy_path, tile_array)

            # Save associated annotations, if there are any
            points_in_tile = point_locs_by_tile[(i, j, k)]
            points_json_path = os.path.join(this_tile_dir, 'points.json')

            if overwrite or not os.path.isfile(points_json_path):
                write_points_to_json(
                    points_json_path, 
                    points_in_tile
                )
                if len(points_in_tile) > 0:
                    tqdm.write(f'Found {len(point_locs_by_tile[(i, j, k)])} point(s) in {tile_name}, saved to points.json')
            
            pbar.update()

        # for indices in tqdm(all_tile_indices, desc='Saving tiles'):
        with tqdm(total=len(all_tile_indices), desc='Saving tiles') as pbar:
            with ThreadPoolExecutor(max_workers=1) as executor:
                tqdm.write(f'Working with up to {executor._max_workers} workers')
                tqdm.write(f'Saving to {self.tiles_dir}')
                futures = [executor.submit(save_tile, idx, pbar) for idx in all_tile_indices]
                # Wait for each future to finish to avoid premature program termination
                for f in futures:
                    f.result()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--index", type=int, required=True)
    # parser.add_argument(
    #     "--data-path", 
    #     type=str,
    #     default="/home/mward19/nobackup/autodelete/fm-data-2"
    # )
    # args = parser.parse_args()

    # dirs = sorted(
    #     d for d in os.listdir(args.data_path)
    #     if d.startswith("tomo-")
    # )

    # target = os.path.join(args.data_path, dirs[args.index])

    # tiler = TomoTiles(target)
    # tiler.tile_tomogram_points(overwrite=True, tiles_dir='tiles-overlapped')
    target = '/home/mward19/nobackup/autodelete/fm-data-2/tomo-10233'
    tiler = TomoTiles(target)
    tiler.tile_tomogram_points(overwrite=True)
