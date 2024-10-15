import os
import hydra
import torch
import numpy as np
from torchvision.models.segmentation import (
    deeplabv3_resnet50, 
    DeepLabV3_ResNet50_Weights, 
)
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from codes.dataset.midog2021_dataset import MIDOG2021Dataset, midog_collate_fn
from codes.framework.framework import FDASegmentationModule
from codes.loss.dice import DiceLoss
from codes.logger.incl import InclLogger
from lightning.pytorch import loggers as pl_loggers
from codes.constant import MITOTIC_CELL_DISTANCE_CUT_OFF
from codes.utils import find_mitotic_cells_from_heatmap, save_visualization
from tqdm import tqdm


CHECKPOINT_DIR = "checkpoints"
BEST_CHECKPOINT_FILENAME = "best"
LAST_CHECKPOINT_FILENAME = "last"


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
        training=False,
    )

    # Setup model
    model = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    model.classifier[4] = torch.nn.Conv2d(256, cfg.model.num_classes, kernel_size=(1, 1))
    model.aux_classifier = None

    # Setup loss
    supervised_loss = DiceLoss(gamma=cfg.loss.gamma)
    consistency_loss = DiceLoss(gamma=cfg.loss.gamma)

    # Instantiate lightning module
    lightning_model = FDASegmentationModule(
        model=model, 
        trn_dataset=trn_dataset,
        val_dataset=val_dataset,
        num_samples_per_epoch=cfg.dataset.num_samples_per_epoch,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        supervised_loss=supervised_loss, 
        consistency_loss=consistency_loss, 
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # Define the loggers
    incl_logger = InclLogger()
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="tb_logs/")

    # A callback for saving the best model based on validation metric
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=BEST_CHECKPOINT_FILENAME,
        verbose=True,
        monitor="F1",
        mode="max",
    )

    # A callback for saving the last model
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=LAST_CHECKPOINT_FILENAME,
        verbose=False,
    )

    # Define the trainer and start training
    trainer = Trainer(
        max_epochs=cfg.trainer.max_epochs,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        devices=cfg.trainer.devices, 
        logger=[incl_logger, tb_logger],
        accelerator="gpu",
        num_sanity_val_steps=0,
        callbacks=[best_checkpoint_callback, last_checkpoint_callback],
    )
    trainer.fit(lightning_model, ckpt_path=f"{CHECKPOINT_DIR}/{LAST_CHECKPOINT_FILENAME}.ckpt")

    # Save visualization with best model
    if cfg.trainer.save_vis and trainer.is_global_zero:
        # Load best model
        best_model_path = f"{trainer._default_root_dir}/{CHECKPOINT_DIR}/{BEST_CHECKPOINT_FILENAME}.ckpt"
        best_state_dict = torch.load(best_model_path, weights_only=True)["state_dict"]
        best_state_dict = {k.replace("model.", ""): v for k, v in best_state_dict.items()}
        model.load_state_dict(best_state_dict, strict=True)
        model.eval()
        print(f"Best model loaded from: {best_model_path}")

        # Save visualization examples
        with torch.no_grad():
            print("Saving visualization ...")
            vis_path = f"{trainer._default_root_dir}/vis"
            os.makedirs(vis_path, exist_ok=True)
            for i, sample in enumerate(tqdm(val_dataset)):
                img = sample["img"]
                pred = model(img.unsqueeze(dim=0))["out"].squeeze(dim=0)
                pred_softmax = pred.softmax(dim=0).detach().cpu().numpy()
                pred_coords, _ = find_mitotic_cells_from_heatmap(
                    pred_softmax, 
                    min_distance=MITOTIC_CELL_DISTANCE_CUT_OFF,
                )
                gt_coords = np.array(sample["gt_coords"])
                save_path = f"{vis_path}/{i}.jpg"
                save_visualization(img, pred_coords, gt_coords, save_path)


if __name__ == "__main__":
    main()
