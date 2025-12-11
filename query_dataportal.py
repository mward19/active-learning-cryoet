"""
- Download and cache data from CryoET Data Portal
- Chunk into subtomograms
- Make annotations accessible with torch.Dataset
"""
# from cryoet_data_portal import Dataset as DP_Dataset
from cryoet_data_portal import Client, Annotation, Tomogram
from tqdm.auto import tqdm
import numpy as np

import contextlib
import sys

from collections import defaultdict
import csv
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

        # Save annotation
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

def get_tomo_ids_in_dataset(dataset_ids, objects, filepath=None):
    client = Client()
    annotations = sum(
        (Annotation.find(
            client, 
            [
                Annotation.run.dataset_id == int(id_), 
                Annotation.object_name == obj
            ]) 
            for id_ in dataset_ids for obj in objects
        ),
        start=[]
    )
    if len(annotations) == 0:
        raise Exception(f'Found 0 annotations corresponding to {object}. Exiting')
    
    # Collect tomogram ids
    ids = []
    for ann in tqdm(annotations, desc='Collecting tomogram IDs'):
        all_tomograms = ann.run.tomograms
        if len(all_tomograms) != 1:
            raise ValueError(f'Annotation {ann.id} has more than 1 associated tomogram (it has {len(all_tomograms)})')
        tomogram = all_tomograms[0]

        # Collect ids
        ids.append(int(tomogram.id))

    if filepath is not None:
        with open(filepath, 'w') as f:
            f.write(f"Tomogram IDs\n")
            for id_ in ids:
                f.write(f"{id_}\n")
    return ids

def get_all_dataset_names_and_lengths(objects: list[str], filepath=None):
    client = Client()
    annotations = sum(
        (Annotation.find(client, [Annotation.object_name == obj]) for obj in objects),
        start=[]
    )
    
    if len(annotations) == 0:
        raise Exception(f'Found 0 annotations corresponding to {objects}. Exiting')
    
    # Map dataset names to number of tomograms
    dataset_counts = {}
    dataset_names = {}
    for ann in tqdm(annotations, desc='Collecting dataset names'):
        dataset_name = ann.run.dataset.title
        dataset_id = ann.run.dataset.id
        num_tomos = len(ann.run.tomograms)
        dataset_counts[dataset_id] = dataset_counts.get(dataset_id, 0) + num_tomos
        if dataset_id not in dataset_names:
            dataset_names[dataset_id] = dataset_name

    if filepath is not None:
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Dataset ID', 'Dataset Name', 'Num Tomograms'])
            for dataset_id in dataset_counts.keys():
                name = dataset_names[dataset_id]
                count = dataset_counts[dataset_id]
                writer.writerow([dataset_id, name, count])
    return dataset_names, dataset_counts

def download_tomo_ids():
    objects = ['bacterial-type flagellum motor']

    from dataloader import get_train_val_dataset_ids
    train_ids, val_ids = get_train_val_dataset_ids()

    print('Fetching IDs')
    get_tomo_ids_in_dataset(val_ids, objects, './data_info/val_tomo_ids.csv')
    get_tomo_ids_in_dataset(train_ids, objects, './data_info/train_tomo_ids.csv')
    

if __name__ == '__main__':
    # objects = ['bacterial-type flagellum motor']
    download_tomo_ids()