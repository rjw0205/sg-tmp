from typing import Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision.models import get_model
from torchvision.models.feature_extraction import (
    create_feature_extractor,
)
from torchvision.transforms import v2
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import TQDMProgressBar

from codes.analysis.analysis import get_image_paths, get_image_name


class PairedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        scanner_images,
        normalize_mean,
        normalize_std,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.scanner_images = scanner_images
        self.data_list = self._build_data_list()

        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=normalize_mean, std=normalize_std)])

    def _build_data_list(self):
        data_list = []
        for scanner, image_paths in self.scanner_images.items():
            for image_path in image_paths:
                image_name = get_image_name(image_path)
                data_item = (scanner, image_name, image_path)
                data_list.append(data_item)
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, data_idx):
        scanner, image_name, image_path = self.data_list[data_idx]

        image = Image.open(image_path)
        image = self.to_tensor(image)

        metadata = {
            "scanner": scanner,
            "image_name": image_name,
        }

        return image, metadata


class FeatureExtractor(LightningModule):
    def __init__(
        self,
        name: str,
        config: dict[str, Any],
        state_path: str,
        extractor_return_node: str,
        output_dir: str,
    ):
        super().__init__()

        self.output_dir = output_dir

        self.model = get_model(name, **config)

        if state_path is not None:
            state_dict = torch.load(state_path)
            self.model.load_state_dict(state_dict, strict=False)

        self.extractor_return_node = extractor_return_node
        self.extractor = create_feature_extractor(
            self.model,
            train_return_nodes={self.extractor_return_node: self.extractor_return_node},
            eval_return_nodes={self.extractor_return_node: self.extractor_return_node},
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.extractor(x)[self.extractor_return_node]
        x = self.avgpool(x)
        return x

    def predict_step(self, batch, batch_idx):
        inputs, metadata = batch
        features = self(inputs)
        features = features.detach().cpu().numpy()

        for idx, feature in enumerate(features):
            scanner = metadata["scanner"][idx]
            image_name = metadata["image_name"][idx]

            scanner_dir = Path(self.output_dir) / scanner
            scanner_dir.mkdir(exist_ok=True, parents=True)

            feature_path = scanner_dir / f"{image_name}.npy"
            np.save(feature_path, feature)

        return features


if __name__ == "__main__":
    data_dir = "/lunit/data/onco/scope_sg/240409"
    scanner_images = get_image_paths(data_dir)

    extractor_return_node = "layer1"

    # extractor_state_path = "bt_rn50_ep200.torch"
    # config = {}
    # output_dir = f"bt_rn50_ep200_{extractor_return_node}_features"
    # normalize_mean = [0.70322989, 0.53606487, 0.66096631]
    # normalize_std = [0.21716536, 0.26081574, 0.20723464]

    extractor_state_path = None
    config = {"weights": "DEFAULT"}
    output_dir = f"imagenet_rn50_{extractor_return_node}_features"
    normalize_mean = (0.485, 0.456, 0.406)
    normalize_std = (0.229, 0.224, 0.225)

    dataset = PairedDataset(scanner_images, normalize_mean=normalize_mean, normalize_std=normalize_std)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, num_workers=7)

    feature_extractor = FeatureExtractor(
        name="resnet50",
        config=config,
        state_path=extractor_state_path,
        extractor_return_node=extractor_return_node,
        output_dir=output_dir
    )
    feature_extractor.eval()

    trainer = Trainer(
        accelerator="auto",
        callbacks=[TQDMProgressBar()],
    )

    trainer.predict(
        feature_extractor,
        dataloaders=dataloader,
        return_predictions=False,
    )
