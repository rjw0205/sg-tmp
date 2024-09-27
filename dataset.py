import torch
from torch.utils.data import Dataset
from utils import read_img, read_wkt


class MIDOG2021Dataset(Dataset):
    def __init__(self, root_path, scanners, transform=None):
        self.root_path = root_path
        self.scanners = scanners
        self.transform = transform
        self.img_paths = self.get_img_paths(root_path)
        self.wkt_paths = [path.replace(".jpg", ".wkt") for path in self.img_paths]

    def get_img_paths(self, root_path):
        img_paths = []
        for scanner in self.scanners:
            img_paths += sorted(glob.glob(f"{root_path}/{scanner}/*/*.jpg"))

        return img_paths
    
    def read_wkt_label(self, wkt_path):
        with open(wkt_path, "r") as f:
            wkt_label = f.read().splitlines()
            assert isinstance(wkt_label, list)

        return wkt_label

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        wkt_path = self.wkt_paths[idx]

        img = read_img(img_path)
        wkt_label = read_wkt(wkt_path)

        return img, wkt_label