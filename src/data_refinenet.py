import random
import json
import math
import os

import cv2
import numpy as np

from torch.utils.data.dataset import Dataset
from transformations import Transformation
from models.model_utils import pre_bgr_image, corner_sub_pix
from dataclasses import replace
from numba import njit
from gridwindow.grid_window import MagicGrid
import torch

def custom_collate(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Return None or handle this case appropriately
    return torch.utils.data.dataloader.default_collate(batch)

@njit(cache=True)
def _add_gaussian(detected_corners_map, x, y, stride, sigma):
    n_sigma = 4
    tl = [int(x - n_sigma * sigma), int(y - n_sigma * sigma)]
    tl[0] = max(tl[0], 0)
    tl[1] = max(tl[1], 0)

    br = [int(x + n_sigma * sigma), int(y + n_sigma * sigma)]
    map_h, map_w = detected_corners_map.shape
    br[0] = min(br[0], map_w * stride)
    br[1] = min(br[1], map_h * stride)

    shift = stride / 2 - 0.5
    for map_y in range(tl[1] // stride, br[1] // stride):
        for map_x in range(tl[0] // stride, br[0] // stride):
            d2 = (map_x * stride + shift - x) * (map_x * stride + shift - x) + \
                (map_y * stride + shift - y) * (map_y * stride + shift - y)
            exponent = d2 / 2 / sigma / sigma
            if exponent > 4.6052:  # threshold, ln(100), ~0.01
                continue
            detected_corners_map[map_y, map_x] += math.exp(-exponent)
            if detected_corners_map[map_y, map_x] > 1:
                detected_corners_map[map_y, map_x] = 1


def create_sample(image: np.ndarray, up_factor, true_corners: tuple):
    # Construct corner label
    w_half = (192 + 64) // (2 * up_factor)
    # up-factor =4 -> w_half = 32
    center_x = int(np.round(true_corners[0]))
    center_y = int(np.round(true_corners[1]))
    print(true_corners)
    print(center_x,center_y)
    

    # Take a patch around the true corner
    patch_og_res = image[center_y - w_half:center_y + w_half,
                         center_x - w_half:center_x + w_half]
    print(patch_og_res.shape)
    # 64x64
    if not patch_og_res.shape == (256 // up_factor, 256 // up_factor, 3):
        return None, None, None

    # Upscale the patch
    patch_up = cv2.resize(patch_og_res, (192 + 64, 192 + 64), cv2.INTER_CUBIC) # resolution 4 times higher
    # print('patch up shape: ', patch_up.shape)    
    # We already know the true corner position, so no need for sub-pixel refinement
    ref_corner = np.array([patch_up.shape[1] // 2, patch_up.shape[0] // 2])
    #print('up factor' , up_factor)
    corr_x = ((true_corners[0] - center_x) * up_factor)
    corr_y = ((true_corners[1] - center_y) * up_factor)
    #print(corr_x,corr_y)
   
    #print('Before correction: ', ref_corner)
    #ref_corner[0]+=corr_x
    #ref_corner[1]+=corr_y

    #print('After correction: ', ref_corner)
    #corr_x=int(np.round(corr_x))
    #corr_y=int(np.round(corr_y))
    # Apply random offset for augmentation
    tl = 32
    #print('Correction:',(corr_x,corr_y))
    off_x = random.randint(-tl+1 + np.sign(corr_x)*np.ceil(np.abs(corr_x)), tl - 1 + np.sign(corr_x)*np.ceil(np.abs(corr_x)))  # random.randint includes BOTH endpoints
    off_y = random.randint(-tl+1 + np.sign(corr_y)*np.ceil(np.abs(corr_y)), tl - 1 + np.sign(corr_y)*np.ceil(np.abs(corr_y)))
    #print('offset: ',(off_x,off_y))
    new_center_x = patch_up.shape[1]//2 + off_x
    new_center_y = patch_up.shape[0]//2 + off_y
    #print('New center: ', (new_center_x,new_center_y))
    patch_new = patch_up[new_center_y - 96:new_center_y + 96,
                         new_center_x - 96:new_center_x + 96]
    #print('Patch_new_shape: ',patch_new.shape)
    patch = cv2.resize(patch_new, (24, 24), cv2.INTER_AREA)
    #print(patch.shape)
    # The true corner position in the patch
    corner_x =  -off_x + tl + corr_x # Adjust for offset
    corner_y =  -off_y + tl + corr_y  
    print(corner_x,corner_y)
    assert 0 <= corner_x < 64 and 0 <= corner_y < 64

    #corner = (int(corner_x),int(corner_y))
    # Generate heatmap for the true corner
    heat = np.zeros((64, 64), dtype=np.float32)
    _add_gaussian(heat, corner[0], corner[1], 1, 2)
    #
    corner = (int(corner_x), int(corner_y))
    return patch, heat, corner



class RefineDataset(Dataset):
    def __init__(self, configs, labels, images_folder, validation=False, visualize=False, total=4):
        super().__init__()
        self.s_factor = 2
        self.zoom = 1
        self.total = total
        configs = replace(configs, input_size=(4112 * self.s_factor * self.zoom,
                                               2176 * self.s_factor * self.zoom))
        self._images_folder = images_folder
        self._visualize = visualize
        with open(labels, 'r') as f:
            self.labels = json.load(f)
        #print(self.labels)
        # Read from json file
        self.image_paths = [label["image_file"] for label in self.labels]
        self.true_corners = [label["true"] for label in self.labels]
        #self.detected_corners = [label["detected"] for label in self.labels]

        seed = 42 if validation else None
        if seed is not None:
            random.seed(seed)
        #self.transform = Transformation(configs, negative_p=0, refinenet=True, seed=seed)

    def __getitem__(self, idx):
        if idx >= len(self.labels):
            return
        label = self.labels[idx]
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
       # detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        print('Image: ', label['image_file'])
        try:
        # Try reading the image
            #print('reading')
            image = cv2.imread(os.path.join(self._images_folder, label['image_file']), cv2.IMREAD_COLOR)
            
            # If the image is None, the path might be invalid or the file is unreadable
            if image is None:
                print(f"Unable to read image at { label['image_file']}. Skipping...")
                self.labels.pop(idx)
                return 
            gray = cv2.imread(os.path.join(self._images_folder, label['image_file']), cv2.IMREAD_GRAYSCALE)
            detected_corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
           
            #return image
            # Apply pipeline of transformations
            #print('still reading')
            #image, self.detected_corners, *_ = self.transform(image).values() # output an image of the train with the chessboard attached

            patch_resized = []  # Just for visualization

            heatmaps = []
            patches = []
            up_factor = 16 // self.s_factor
            #random.shuffle(self.detected_corners)
            if len(detected_corners) > 0:
                #print('detected')   
                for true_corner in self.true_corners[idx]:
                    patch, heat, corner = create_sample(image, up_factor, true_corner)
                # Pad not implemented, sometimes there are regions not big enough, it's not a problem
                    if patch is None:
                        continue
                
                    patches.append(patch)
                    heatmaps.append(heat)
                # Just visualization
                    if self._visualize:
                    # Visualization in original resolution is poor, visualize in 8x patch!
                        patch_vis = cv2.resize(patch, (192, 192), cv2.INTER_CUBIC)
                        corner_vis = (corner[0] + 64, corner[1] + 64)
                        patch_vis = cv2.circle(patch_vis, (corner_vis[0], corner_vis[1]),
                                        radius=5, color=(0, 255, 0),
                                        thickness=2)

                        patch_vis = cv2.circle(patch_vis, (patch_vis.shape[1] // 2, patch_vis.shape[0]// 2),
                                        radius=5, color=(0, 0, 255),
                                        thickness=2)
                        patch_resized.append(patch_vis)

                    if len(patches) == self.total:
                        break  # Done!

                if self._visualize:
                    w = MagicGrid(640, 640, waitKey=0, draw_outline=True)
                    heatmaps_vis = [(h * 255).astype(np.uint8) for h in heatmaps]
                    if w.update([*patch_resized, *heatmaps_vis]) == ord('q'):
                        import sys
                        sys.exit()

                patches = [pre_bgr_image(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)) for p in patches]
                heatmaps = [h[None, ...] for h in heatmaps]
            else:
    # If no patches, generate fallback default patches
                if len(patches) == 0:
                    default_patch = np.zeros((24, 24), dtype=np.uint8)  # Default blank patch
                    default_heatmap = np.zeros((64, 64), dtype=np.float32)  # Default blank heatmap
                    _add_gaussian(default_heatmap, 32, 32, 1, 2)  # Centered Gaussian

        # Add fallback patches to maintain batch consistency
                    for _ in range(self.total):
                        patches.append(pre_bgr_image(default_patch))  # Grayscale default patch
                        heatmaps.append(default_heatmap[None, ...])  # Add channel dimension

# If not enough, duplicate samples to reach required total
            while len(patches) < self.total:
                idx = random.randint(0, len(patches) - 1)
                patches.append(patches[idx])
                heatmaps.append(heatmaps[idx])
            # Must have same length for each batch.
            # We duplicate samples to reach correct numbering
            # Training with total < 16, this should not happen
            #missing = self.total - len(heatmaps)
            #print(missing)
#            for _ in range(missing):
#                idx = random.randint(0, len(patches) - 1)
#                patches.append(patches[idx])
#                heatmaps.append(heatmaps[idx])

            return patches, heatmaps
        
        except Exception as e:
            # Handle any exceptions, such as file not found or read errors
            print(f"Error reading {label['image_file']}: {e}. Skipping...")
            #self.labels.pop(idx)
            return 
        
       
        

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import configs
    from configs import load_configuration
    config = load_configuration(configs.CONFIG_PATH) #ok
    dataset = RefineDataset(config,
                            config.train_labels,
                            config.train_images,
                            visualize=True,
                            validation=False)

    dataset_val = RefineDataset(config,
                                config.val_labels,
                                config.val_images,
                                visualize=False,
                                validation=True)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,collate_fn = custom_collate)
    val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0,collate_fn = custom_collate)

    for i, b in enumerate(train_loader):
        print(i)

    for i, b in enumerate(val_loader):
        print(i)
