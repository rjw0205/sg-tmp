import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50, 
    DeepLabV3_ResNet50_Weights, 
) 
from pytorch_lightning import Trainer
from codes.dataset.midog2021_dataset import MIDOG2021Dataset, midog_collate_fn
from codes.framework.framework import FDASegmentationModule


# Setup datasets
DATA_ROOT_PATH = "/lunit/data/midog_2021_patches"
trn_dataset = MIDOG2021Dataset(
    root_path=DATA_ROOT_PATH, 
    scanners=["Aperio_CS2", "Hamamatsu_S360", "Leica_GT450"],  # Leica_GT450 is not annotated
    training=True, 
    do_fda=True,
)
val_dataset = MIDOG2021Dataset(
    root_path=DATA_ROOT_PATH, 
    scanners=["Hamamatsu_XR"], 
    training=False, 
    do_fda=False,
)

# Setup model
num_classes = 3  # BG, Mitotic-Figure, Non-Miototic-Figure
model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=(1, 1))
model.aux_classifier = None

# Setup loss
supervised_loss = torch.nn.CrossEntropyLoss()  # TODO
consistency_loss = torch.nn.CrossEntropyLoss()  # TODO

# Instantiate lightning module
lr = 1e-4
batch_size = 4
num_workers = 8
lightning_model = FDASegmentationModule(
    model, 
    trn_dataset,
    val_dataset,
    batch_size,
    num_workers,
    supervised_loss, 
    consistency_loss, 
    lr,
)

# Define the trainer and start training
trainer = Trainer(
    max_epochs=10,
    check_val_every_n_epoch=5,
    log_every_n_steps=10,
    devices=4, 
    accelerator="gpu",
)
trainer.fit(lightning_model)
