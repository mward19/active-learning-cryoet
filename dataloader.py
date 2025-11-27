"""
- Download and cache data from CryoET Data Portal
- Chunk into subtomograms
- Make annotations accessible with torch.Dataset
"""
# from cryoet_data_portal import Dataset as DP_Dataset
from cryoet_data_portal import Client, Annotation, Tomogram
# import ome_zarr as zarr
from tqdm.auto import tqdm

import contextlib
import sys
from collections import defaultdict
from typing import List
import os
import json

def hide_output(func, *args, **kwargs):
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            return func(*args, **kwargs)# 

def save_tomos_with_obj(objects: List[str], save_dir, ensure_newest=False):
    # image_dir = os.path.join(save_dir, 'images')
    # labels_path = os.path.join(save_dir, 'labels.json')

    client = Client()
    annotations = sum(
        (Annotation.find(client, [Annotation.object_name == obj]) for obj in objects),
        start=[]
    )
    # annotations = annotations[:2]

    if len(annotations) == 0:
        raise Exception(f'Found 0 annotations corresponding to {object}. Exiting')

    for ann in tqdm(annotations, desc='Downloading tomograms'):
        all_tomograms = ann.run.tomograms
        if len(all_tomograms) != 1:
            raise ValueError(f'Annotation {ann.id} has more than 1 associated tomogram (it has {len(all_tomograms)})')
        tomogram = all_tomograms[0]

        # Save tomogram if not saved already
        tomogram_dir = os.path.join(save_dir, f'tomo-{tomogram.id}')
        zarr_path = os.path.join(tomogram_dir, f'{tomogram.name}.zarr')
        if not os.path.exists(zarr_path): # Not path.isfile because zarr is technically a dir
            tqdm.write(f'Saving tomogram {tomogram.name} as {zarr_path}')
            hide_output(tomogram.download_omezarr, tomogram_dir)
        else:
            tqdm.write(f'Already saved tomogram {tomogram.name} at {zarr_path}')

        has_ndjson = any(f.endswith(".ndjson") for f in os.listdir(tomogram_dir))
        if ensure_newest or not has_ndjson:
            tqdm.write(f'Saving annotation {ann.id}')
            try:
                hide_output(ann.download, dest_path=tomogram_dir, format='ndjson')
            except KeyError as e:
                pass
        else:
            tqdm.write(f'Found ndjson in {tomogram_dir}, assuming annotation {ann.id} is already saved')

        # Save tomogram info
        tomo_metadata_path = os.path.join(tomogram_dir, 'tomo-metadata.json')
        if not os.path.isfile(tomo_metadata_path): 
            tomo_metadata = {
                'id': tomogram.id,
                'name': tomogram.name,
                'zarr_https': tomogram.https_omezarr_dir,
                'size_x': tomogram.size_x,
                'size_y': tomogram.size_y,
                'size_z': tomogram.size_z,
                'voxel_spacing': tomogram.voxel_spacing,
            }
            with open(tomo_metadata_path, 'w') as f:
                json.dump(tomo_metadata, f)

# def read_zarr(zarr_path):
#     reader = zarr.Reader(zarr_path)
#     print()

# class TomogramSplitter:
#     def __init__(self, image_dir, labels_path):
#         with open(labels_path, 'r') as f:
#             self.labels = json.loads(labels_path)

save_tomos_with_obj(
    ['bacterial-type flagellum motor'],
    '/home/mward19/nobackup/autodelete/fm-data-2'
)
# read_zarr('/home/mward19/nobackup/autodelete/fm-data/images/')