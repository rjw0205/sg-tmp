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


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig):
    # If do FDA, consistency loss weight should be positive number.
    if cfg.dataset.do_fda:
        assert cfg.loss.consistency_loss_weight > 0.0

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
    supervised_loss = DiceLoss(p=1.0, gamma=cfg.loss.gamma)
    consistency_loss = DiceLoss(p=1.0, gamma=cfg.loss.gamma)

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
        consistency_loss_weight=cfg.loss.consistency_loss_weight, 
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay,
        scheduler=cfg.optimizer.scheduler,
        per_class_loss_weight=cfg.loss.per_class_loss_weight,
        max_epoch=cfg.trainer.max_epochs,
    )

    # Define the loggers
    incl_logger = InclLogger()
    tb_logger = pl_loggers.TensorBoardLogger(save_dir="tb_logs/")

    # A callback for saving the best model based on validation metric
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best",
        verbose=True,
        monitor="F1",
        mode="max",
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
        callbacks=checkpoint_callback,
    )
    trainer.fit(lightning_model)

    # Save visualization with best model
    if cfg.trainer.num_save_vis > 0 and trainer.is_global_zero:
        # Load best model
        best_model_path = f"{trainer._default_root_dir}/checkpoints/best.ckpt"
        best_state_dict = torch.load(best_model_path)["state_dict"]
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
                if i >= num_save_vis:
                    break

                img = sample["img"]
                pred = model(img.unsqueeze(dim=0))["out"].softmax(dim=1).squeeze(dim=0)
                pred = pred.detach().cpu().numpy()
                pred_coords, _ = find_mitotic_cells_from_heatmap(
                    pred, min_distance=MITOTIC_CELL_DISTANCE_CUT_OFF,
                )
                gt_coords = np.array(sample["gt_coords"])
                save_path = f"{vis_path}/{i}_pred_{len(pred_coords)}_GT_{len(gt_coords)}.jpg"
                save_visualization(img, pred, pred_coords, gt_coords, save_path)


if __name__ == "__main__":
    main()
