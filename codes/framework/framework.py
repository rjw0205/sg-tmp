import random
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from codes.dataset.midog2021_dataset import midog_collate_fn
from codes.utils import (
    find_mitotic_cells_from_heatmap, 
    compute_tp_and_fp, 
    compute_precision_recall_f1, 
)
from codes.constant import MITOTIC_CELL_DISTANCE_CUT_OFF
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class FDASegmentationModule(pl.LightningModule):
    def __init__(
        self, 
        model, 
        trn_dataset, 
        val_dataset, 
        num_samples_per_epoch, 
        batch_size, 
        num_workers, 
        supervised_loss, 
        consistency_loss, 
        consistency_loss_weight, 
        lr,
        weight_decay,
        scheduler,
        per_class_loss_weight,
        max_epoch,
    ):
        super(FDASegmentationModule, self).__init__()
        self.model = model
        self.trn_dataset = trn_dataset
        self.val_dataset = val_dataset
        self.num_samples_per_epoch = num_samples_per_epoch
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.supervised_loss = supervised_loss
        self.consistency_loss = consistency_loss
        self.consistency_loss_weight = consistency_loss_weight
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.per_class_loss_weight = per_class_loss_weight
        self.max_epoch = max_epoch

        # Store number of GT, TP, FP per image for evaluation
        self.global_num_gt = []
        self.global_num_tp = []
        self.global_num_fp = []

        # Save hyperparameters
        self.save_hyperparameters(
            'lr', 'weight_decay', 'scheduler', 'batch_size', 'per_class_loss_weight'
        )

    def subsample_trn_dataset(self):
        # Subsample indices for training epoch which includes,
        # 1. sample which have 0 annotation 
        # 2. sample which have only MF annotation
        # 3. sample which have only non-MF annotation
        # 4. sample which have both MF and non-MF annotation
        N = self.num_samples_per_epoch // 4
        indices_for_training = []
        indices_for_training += random.sample(self.trn_dataset.indices_no_cell, N)
        indices_for_training += random.sample(self.trn_dataset.indices_only_mf, N)
        indices_for_training += random.sample(self.trn_dataset.indices_only_nmf, N)
        indices_for_training += random.sample(self.trn_dataset.indices_both_mf_nmf, N)
        self.trn_subset = Subset(self.trn_dataset, indices_for_training)

    def on_fit_start(self):
        # Subsample before the first epoch
        self.subsample_trn_dataset()

    def on_train_epoch_start(self):
        # Subsample for every train epoch, including after the first one
        self.subsample_trn_dataset()

    def train_dataloader(self):
        # Use the subsampled training dataset
        return DataLoader(
            self.trn_subset,
            batch_size=self.batch_size,
            shuffle=True, 
            collate_fn=midog_collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        # Use the full validation dataset (no subsampling)
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            collate_fn=midog_collate_fn,
            num_workers=self.num_workers,
            drop_last=True,
        )
    
    def forward(self, x):
        return self.model(x)["out"].softmax(dim=1)

    def training_step(self, batch, batch_idx):
        imgs_dict, seg_labels, gt_coords = batch
        
        # Compute supervised loss
        imgs = imgs_dict["original"]
        preds = self.forward(imgs)
        supervised_loss = self.supervised_loss(preds, seg_labels, self.per_class_loss_weight)

        # Compute consistency loss
        if "fda" in imgs_dict:
            fda_imgs = imgs_dict["fda"]
            fda_preds = self.forward(fda_imgs)
            consistency_loss = self.consistency_loss(preds, fda_preds)
        else:
            consistency_loss = torch.tensor(0.0).cuda()

        # Log loss
        self.log("train_supervised_loss", supervised_loss, sync_dist=True, batch_size=self.batch_size)
        self.log("train_consistency_loss", consistency_loss, sync_dist=True, batch_size=self.batch_size)

        # Log learning rate
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("LR", lr, on_step=True, logger=True)

        # Return losses
        return {
            "loss": supervised_loss + self.consistency_loss_weight * consistency_loss,
            "supervised_loss": supervised_loss,
            "consistency_loss": consistency_loss,
        }

    def validation_step(self, batch, batch_idx):
        imgs_dict, seg_labels, gt_coords = batch

        # Compute supervised loss
        imgs = imgs_dict["original"]
        preds = self.forward(imgs)
        supervised_loss = self.supervised_loss(preds, seg_labels, self.per_class_loss_weight)

        # Log loss
        self.log("val_supervised_loss", supervised_loss, sync_dist=True, batch_size=self.batch_size)

        # Collect predictions and GT for metric calculation
        preds = preds.detach().cpu().numpy()
        for pred, gt_coord in zip(preds, gt_coords):
            gt_coord = np.array(gt_coord)
            pred_coord, pred_score = find_mitotic_cells_from_heatmap(
                pred, min_distance=MITOTIC_CELL_DISTANCE_CUT_OFF,
            )
            num_preds = len(pred_coord)
            num_gt = len(gt_coord)

            # Calculate number of TP and FP
            if num_preds == 0:
                num_tp, num_fp = 0, 0
            elif num_gt == 0:
                num_tp, num_fp = 0, num_preds
            else:
                num_tp, num_fp = compute_tp_and_fp(
                    pred_coord, 
                    pred_score, 
                    gt_coord, 
                    MITOTIC_CELL_DISTANCE_CUT_OFF,
                )

            self.global_num_gt.append(num_gt)
            self.global_num_tp.append(num_tp)
            self.global_num_fp.append(num_fp)

        return {
            "loss": supervised_loss,
            "supervised_loss": supervised_loss,
        }

    def on_validation_epoch_end(self):
        # Gather over GPUs
        all_num_gt = self.all_gather(self.global_num_gt)
        all_num_tp = self.all_gather(self.global_num_tp)
        all_num_fp = self.all_gather(self.global_num_fp)

        # Calculate population GT, TP, FP
        all_num_gt = np.sum([n.cpu().numpy() for n in all_num_gt])
        all_num_tp = np.sum([n.cpu().numpy() for n in all_num_tp])
        all_num_fp = np.sum([n.cpu().numpy() for n in all_num_fp])

        # Calculate metrics
        precision, recall, f1 = compute_precision_recall_f1(all_num_gt, all_num_tp, all_num_fp)
        self.log("Precision", round(100 * precision, 2), sync_dist=True)
        self.log("Recall", round(100 * recall, 2), sync_dist=True)
        self.log("F1", round(100 * f1, 2), sync_dist=True)

        # Refresh cell counts for next evaluation
        self.global_num_gt = []
        self.global_num_tp = []
        self.global_num_fp = []

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler == "plain":
            return optimizer
        elif self.scheduler == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epoch)
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        else:
            raise NotImplementedError()
