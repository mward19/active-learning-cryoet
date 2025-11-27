import numpy as np
from ome_zarr.reader import Reader as ZarrReader
from ome_zarr.io import parse_url

import os
import math

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
        np_array = dask_data[idx_high_res].compute()
        return np_array

    def __init__(self, tomo_dir):
        """Assumes tomo_dir contains one .zarr file and n>0 .ndjson annotation files."""
        # Get zarr file
        zarr_files = [f for f in os.listdir(tomo_dir) if f.endswith(".zarr")]
        if len(zarr_files) != 1:
            raise ValueError(f"Expected exactly one .zarr file, found {len(zarr_files)}")
        self.zarr_file = os.path.join(tomo_dir, zarr_files[0])
        
        # Get annotations
        self.annotation_files = [f for f in os.listdir(tomo_dir) if f.endswith(".ndjson")]

    # def save_tiles(tiles_dir: str='tiles', )


if __name__ == '__main__':
    data_path = '/home/mward19/nobackup/autodelete/fm-data-2'
    TomoTiles(os.path.join(data_path, 'tomo-10434'))

