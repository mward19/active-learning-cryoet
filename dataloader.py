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

def save_tomos_with_obj(objects: List[str], save_dir, overwrite_labels=False):
    image_dir = os.path.join(save_dir, 'images')
    labels_path = os.path.join(save_dir, 'labels.json')

    client = Client()
    annotations = sum(
        (Annotation.find(client, [Annotation.object_name == obj]) for obj in objects),
        start=[]
    )
    # annotations = annotations[:2]

    if len(annotations) == 0:
        raise Exception(f'Found 0 annotations corresponding to {object}. Exiting')

    annotations_by_tomo = defaultdict(list)

    for ann in tqdm(annotations, desc='Fetching metadata'):
        all_tomograms = ann.run.tomograms
        if len(all_tomograms) != 1:
            raise ValueError(f'Annotation {ann.id} has more than 1 associated tomogram (it has {len(all_tomograms)})')
        
        annotations_by_tomo[all_tomograms[0].id].append(ann)
    
    labels = dict()
    for tomo_id, annotations in tqdm(
            annotations_by_tomo.items(), 
            total=len(annotations_by_tomo),
            desc='Loading data'
        ):
        # Get the tomogram corresponding to this ID
        tomograms = Tomogram.find(client, [Tomogram.id == tomo_id])
        if len(tomograms) != 1:
            raise ValueError(f'Tomogram id {tomo_id} has more than 1 associated tomogram (it has {len(tomograms)})')
        tomogram = tomograms[0]

        # Save it to the cache
        image_path = os.path.join(image_dir, f'{tomogram.name}.zarr')
        # Only save tomogram if not saved already
        if not os.path.isfile(image_path):
            tqdm.write(f'Saving tomogram {tomogram.name} as {image_path}')
            hide_output(tomogram.download_omezarr, image_dir)
        else:
            tqdm.write(f'Already saved tomogram {tomogram.name} at {image_path}, skipping')

        # Save all associated annotations in labels
        labels[tomo_id] = {
            'tomo_id': tomo_id,
            'tomo_name': tomogram.name,
            'tomo_path': image_path,
            'zarr_https': tomogram.https_omezarr_dir,
            'ann_ids': [ann.id for ann in annotations],
            'size_x': tomogram.size_x,
            'size_y': tomogram.size_y,
            'size_z': tomogram.size_z,
            'voxel_spacing': tomogram.voxel_spacing,
        }

    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=4)
        tqdm.write(f'Saved labels to {labels_path}')

def read_zarr(zarr_path):
    reader = zarr.Reader(zarr_path)
    print()

class TomogramSplitter:
    def __init__(self, image_dir, labels_path):
        with open(labels_path, 'r') as f:
            self.labels = json.loads(labels_path)

save_tomos_with_obj(
    ['bacterial-type flagellum motor'],
    '/home/mward19/nobackup/autodelete/fm-data'
)
# read_zarr('/home/mward19/nobackup/autodelete/fm-data/images/')