import hydra
import torch
from torchvision.models.segmentation import (
    deeplabv3_resnet50, 
    DeepLabV3_ResNet50_Weights, 
)
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from codes.dataset.midog2021_dataset import MIDOG2021Dataset, midog_collate_fn
from codes.framework.framework import FDASegmentationModule
from codes.loss.dice import DiceLoss


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # Setup dataset
    trn_dataset = MIDOG2021Dataset(
        root_path=cfg.dataset.root_path, 
        scanners=cfg.dataset.trn_scanners, 
        do_fda=cfg.dataset.do_fda,
        training=True,
    )
    val_dataset = MIDOG2021Dataset(
        root_path=cfg.dataset.root_path, 
        scanners=cfg.dataset.val_scanners, 
        do_fda=False,
        training=True,
    )

    # Setup model
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    model.classifier[4] = torch.nn.Conv2d(256, cfg.model.num_classes, kernel_size=(1, 1))
    model.aux_classifier = None

    # Setup loss
    supervised_loss = DiceLoss()
    consistency_loss = DiceLoss()

    # Instantiate lightning module
    lightning_model = FDASegmentationModule(
        model=model, 
        trn_dataset=trn_dataset,
        val_dataset=val_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        supervised_loss=supervised_loss, 
        consistency_loss=consistency_loss, 
        lr=cfg.optimizer.lr,
    )

    # Define the trainer and start training
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        devices=cfg.trainer.devices, 
        accelerator="gpu",
    )
    trainer.fit(lightning_model)


if __name__ == "__main__":
    main()
