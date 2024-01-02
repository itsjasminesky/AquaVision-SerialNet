from natsort import natsorted
from glob import glob
import cv2
import json
import os
import tqdm


def list_image_path(image_dir):
    """Getting list of image paths."""
    ext=['.png', '.jpg', '.PNG', '.JPG', '.jpeg']
    image_paths = []
    for e in ext:
        find_paths = glob(os.path.join(image_dir, f'*{e}'))
        image_paths += find_paths
    image_path_list = natsorted(image_paths)
    return image_path_list


def read_image(image_path):
    """Reading image path."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def write_json(file_name, data_dict):
    with open(file_name, 'w') as f:
        json.dump(data_dict, f, indent=4)
    return None


def resize_image(image_dir, size=2048):
    """Resizing all images in directory."""
    image_path_list = list_image_path(image_dir)
    for image_path in tqdm.tqdm(image_path_list, 'Resizing Image'):
        image = cv2.imread(image_path)
        h, w, _ = image.shape
        
        # FINDING RATIO
        if w > h:
            ratio = size / w
        else:
            ratio = size / h
        
        nh, nw = int(h*ratio), int(w*ratio)
        
        image = cv2.resize(image, (nw, nh))
        
        cv2.imwrite(image_path, image)
    return None