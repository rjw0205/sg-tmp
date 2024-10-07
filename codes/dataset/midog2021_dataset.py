import os
import glob
import torch
import random
import pickle

import numpy as np
import albumentations as A

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, default_collate

from codes.constant import MITOTIC_CELL_CLS_IDX
from codes.fda import fda_augmentation
from codes.utils import read_img, parse_wkt_annotation, draw_segmentation_label


class MIDOG2021Dataset(Dataset):
    def __init__(
        self, 
        root_path, 
        scanners, 
        training, 
        do_fda,
        fda_beta_start=0.001, 
        fda_beta_end=0.01, 
        radius=15,
    ):
        assert os.path.exists(root_path), f"{root_path} does not exist."
        assert isinstance(scanners, list)
        assert isinstance(training, bool)
        assert isinstance(fda_beta_start, float)
        assert isinstance(fda_beta_end, float)
        assert 0.0 < fda_beta_start <= fda_beta_end <= 0.5

        self.root_path = root_path
        self.scanners = scanners
        self.radius = radius
        self.training = training
        self.fda_beta_start = fda_beta_start
        self.fda_beta_end = fda_beta_end
        self.do_fda = do_fda

        self.img_paths = self.get_img_paths(root_path)
        self.img_paths_per_scanner = self.get_img_paths_per_scanner(root_path)
        self.metadata = self.get_metadata(root_path)
        self.transform = self.get_transforms()
        self.setup_indices()
    
    def setup_indices(self):
        self.indices_with_zero_annot = []
        self.indices_with_at_least_one_annot = []

        for i, img_path in enumerate(self.img_paths):
            meta_key = img_path.replace(".jpg", "")
            metadata = self.metadata[meta_key]
            num_valid_cells = metadata["num_mitotic_figure"] + metadata["num_non_mitotic_figure"]
            if num_valid_cells == 0:
                self.indices_with_zero_annot.append(i)
            else:
                self.indices_with_at_least_one_annot.append(i)

    def get_img_paths(self, root_path):
        img_paths = []
        for scanner in self.scanners:
            img_paths += sorted(glob.glob(f"{root_path}/{scanner}/*/*.jpg"))

        return img_paths

    def get_img_paths_per_scanner(self, root_path):
        img_paths_per_scanner = {}
        for scanner in self.scanners:
            img_paths_per_scanner[scanner] = sorted(glob.glob(f"{root_path}/{scanner}/*/*.jpg"))

        return img_paths_per_scanner

    def get_metadata(self, root_path):
        with open(f"{root_path}/metadata.pkl", "rb") as f:
            metadata = pickle.load(f)
            return metadata
    
    def get_transforms(self):
        random_transforms = []
        to_tensor_transforms = [
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0), # 0~255 to 0.0~1.0
            ToTensorV2(),  # HWC np.array to CHW torch.tensor
        ]

        if self.training:
            # Random data augmentations only for training
            random_transforms.append(A.HorizontalFlip(p=0.5))
            random_transforms.append(A.VerticalFlip(p=0.5))
            random_transforms.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.8))
        
        return A.ReplayCompose(
            random_transforms + to_tensor_transforms,
            keypoint_params=A.KeypointParams(format="xy"),
        )

    def random_fda(self, src_img, src_scanner):
        # Randomly choose scanner to be used for FDA
        candidate_scanners = [s for s in self.scanners if s != src_scanner]
        tgt_scanner = random.choice(candidate_scanners)

        # Randomly choose img for FDA from selected scanner
        tgt_img_path = random.choice(self.img_paths_per_scanner[tgt_scanner])
        tgt_img = read_img(tgt_img_path)

        # Perform FDA
        L = np.random.uniform(self.fda_beta_start, self.fda_beta_end)
        fda_img = fda_augmentation(src_img, tgt_img, channel="V", L=L)
        return fda_img

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        wkt_path = img_path.replace(".jpg", ".wkt")

        # Read image and create FDA image
        img = read_img(img_path)
        if self.do_fda:
            src_scanner = self.metadata[img_path.replace(".jpg", "")]["scanner"]
            fda_img = self.random_fda(img, src_scanner)

        # Read wkt annotation
        gt_coords, gt_categories = parse_wkt_annotation(wkt_path)

        # Apply transform to image and points
        t = self.transform(image=img, keypoints=gt_coords)
        img, gt_coords = t["image"], t["keypoints"]

        # Create a segmentation label
        seg_label = draw_segmentation_label(img, gt_coords, gt_categories, self.radius)

        # Exclude non-mitotic cells from GT points, which are not used for metric computation.
        gt_coords = [
            gt_coord for gt_coord, gt_cls in zip(gt_coords, gt_categories) 
            if gt_cls == MITOTIC_CELL_CLS_IDX
        ]

        # Output of dataset
        sample = {"img": img, "seg_label": seg_label,  "gt_coords": gt_coords}

        # Apply the same transform to FDA image
        if self.do_fda:
            t_fda = A.ReplayCompose.replay(t["replay"], image=fda_img)
            fda_img = t_fda["image"]
            sample["fda_img"] = fda_img

        return sample


def midog_collate_fn(batch):
    # Stack images
    imgs = [item["img"] for item in batch]
    imgs = torch.stack(imgs, dim=0)
    imgs_dict = {"original": imgs}

    # Stack FDA images
    if "fda_img" in batch[0]:
        fda_imgs = [item["fda_img"] for item in batch]
        fda_imgs = torch.stack(fda_imgs, dim=0)
        imgs_dict["fda"] = fda_imgs

    # Stack segmentation labels
    seg_labels = [item["seg_label"] for item in batch]
    seg_labels = torch.stack(seg_labels, dim=0)

    # Return GT point as is, since each sample has different number of points
    gt_coords = [item["gt_coords"] for item in batch]

    return imgs_dict, seg_labels, gt_coords


if __name__ == "__main__":
    root_path = "/lunit/data/midog_2021_patches"
    scanners = ["Aperio_CS2", "Hamamatsu_S360", "Hamamatsu_XR"]
    training = True
    do_fda = True
    batch_size = 1

    # Sanity check
    dataset = MIDOG2021Dataset(root_path, scanners, training, do_fda)
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=midog_collate_fn)
    for imgs_dict, seg_labels, gt_coords in data_loader:
        print("original img shape:", imgs_dict["original"].shape)
        if do_fda:
            print("FDA img shape:", imgs_dict["fda"].shape)
        print("Segmentation label shape:", seg_labels.shape)
        print("GT coordinates:", gt_coords)
        break
