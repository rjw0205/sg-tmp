import os
import glob
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from utils import read_img, parse_wkt_annotation, draw_segmentation_label


class MIDOG2021Dataset(Dataset):
    def __init__(self, root_path, scanners, training, radius=15):
        assert os.path.exists(root_path), f"{root_path} does not exist."
        assert isinstance(scanners, list)
        assert isinstance(training, bool)
        self.root_path = root_path
        self.scanners = scanners
        self.radius = radius
        self.training = training
        self.img_paths = self.get_img_paths(root_path)
        self.transform = self.get_transforms()
        self.wkt_paths = [path.replace(".jpg", ".wkt") for path in self.img_paths]

    def get_img_paths(self, root_path):
        img_paths = []
        for scanner in self.scanners:
            img_paths += sorted(glob.glob(f"{root_path}/{scanner}/*/*.jpg"))

        return img_paths
    
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

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = read_img(self.img_paths[idx])
        fda_img = img.copy() # TODO
        gt_points, gt_categories = parse_wkt_annotation(self.wkt_paths[idx])

        # Apply transform to image and points
        t = self.transform(image=img, keypoints=gt_points)
        img, gt_points = t["image"], t["keypoints"]

        # Apply the same transform to FDA image
        t_fda = A.ReplayCompose.replay(t["replay"], image=fda_img)
        fda_img = t_fda["image"]

        # Create a segmentation label
        seg_label = draw_segmentation_label(img, gt_points, gt_categories, self.radius)

        return img, fda_img, seg_label, gt_points


if __name__ == "__main__":
    root_path = "/storage6/pp/share/rjw0205/scanner_generalization/paper_submission/midog_2021_patches"
    scanners = ["Aperio_CS2", "Hamamatsu_S360", "Hamamatsu_XR", "Leica_GT450"]
    training = True
    dataset = MIDOG2021Dataset(root_path, scanners, training)
    t_img, t_fda_img, seg_label, t_gt_points = dataset[0]
