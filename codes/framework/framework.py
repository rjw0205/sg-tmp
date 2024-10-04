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


class FDASegmentationModule(pl.LightningModule):
    def __init__(
        self, 
        model, 
        trn_dataset, 
        val_dataset, 
        batch_size,
        num_workers,
        supervised_loss, 
        consistency_loss, 
        lr,
    ):
        super(FDASegmentationModule, self).__init__()
        self.model = model
        self.trn_dataset = trn_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.supervised_loss = supervised_loss
        self.consistency_loss = consistency_loss
        self.lr = lr

        # Store predictions and GT for each batch
        self.cut_off = 30
        self.predictions = []
        self.gt_coords = []

    def subsample_trn_dataset(self):
        # Subsample new indices for each epoch of training it includes,
        # - indices which contains cell annotations
        # - indices which don't have cell annotation (randomly sampled)
        indices_for_training = []
        indices_for_training += self.trn_dataset.indices_with_at_least_one_annot
        indices_for_training += random.sample(
            self.trn_dataset.indices_with_zero_annot, 
            len(self.trn_dataset.indices_with_at_least_one_annot)
        )
        self.trn_subset = Subset(self.trn_dataset, indices_for_training)
        # self.trn_subset = Subset(self.trn_dataset, list(range(32)))

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
        )

    def val_dataloader(self):
        # Use the full validation dataset (no subsampling)
        return DataLoader(
            # Subset(self.val_dataset, list(range(1024))),
            self.val_dataset, 
            batch_size=self.batch_size,
            shuffle=False, 
            collate_fn=midog_collate_fn,
            num_workers=self.num_workers,
        )
    
    def forward(self, x):
        return self.model(x)["out"]

    def training_step(self, batch, batch_idx):
        imgs_dict, seg_labels, gt_coords = batch
        
        # Compute supervised loss
        imgs = imgs_dict["original"]
        preds = self.forward(imgs)
        supervised_loss = self.supervised_loss(preds, seg_labels)

        # Compute consistency loss
        if "fda" in imgs_dict: 
            fda_imgs = imgs_dict["fda"]
            fda_preds = self.forward(fda_imgs)
            consistency_loss = self.consistency_loss(preds, fda_preds)
        else:
            consistency_loss = torch.tensor(0.0)

        # Log loss
        self.log("train_supervised_loss", supervised_loss, sync_dist=True)
        self.log("train_consistency_loss", consistency_loss, sync_dist=True)

        # Return losses
        return {
            "loss": supervised_loss + consistency_loss,
            "supervised_loss": supervised_loss,
            "consistency_loss": consistency_loss,
        }

    def validation_step(self, batch, batch_idx):
        imgs_dict, seg_labels, gt_coords = batch

        # Compute supervised loss
        imgs = imgs_dict["original"]
        preds = self.forward(imgs)
        supervised_loss = self.supervised_loss(preds, seg_labels)

        self.log("val_supervised_loss", supervised_loss, sync_dist=True)

        # Collect predictions and GT for metric calculation
        self.predictions.append(preds.softmax(dim=1).detach().cpu().numpy())
        self.gt_coords += gt_coords

        return {
            "loss": supervised_loss,
            "supervised_loss": supervised_loss,
        }

    def on_validation_epoch_end(self):
        self.predictions = np.concatenate(self.predictions, axis=0)
        assert len(self.predictions) == len(self.gt_coords)

        global_num_gt = 0
        global_num_tp = 0
        global_num_fp = 0

        for i in range(len(self.predictions)):
            # Collect num GT
            gt_coords = np.array(self.gt_coords[i])
            num_gt = len(gt_coords)
            global_num_gt += num_gt

            # Find mitotic cells from prediction heatmap
            pred = self.predictions[i]
            pred_coords, pred_scores = find_mitotic_cells_from_heatmap(pred)
            num_preds = len(pred_coords)

            # Calculate TP and FP
            if num_preds == 0:
                # If no preidcted cells, continue
                continue
            elif num_gt == 0:
                # If no GT cells, all predicted cells are False Positive
                global_num_fp += num_preds
                continue
            else:
                num_tp, num_fp = compute_tp_and_fp(
                    pred_coords, 
                    pred_scores, 
                    gt_coords, 
                    self.cut_off,
                )
                global_num_tp += num_tp
                global_num_fp += num_fp

        precision, recall, f1 = compute_precision_recall_f1(
            global_num_gt, 
            global_num_tp, 
            global_num_fp,
        )
        self.log("Precision", round(precision, 4), sync_dist=True)
        self.log("Recall", round(recall, 4), sync_dist=True)
        self.log("F1", round(f1, 4), sync_dist=True)

        # Refresh for next evaluation
        self.predictions = []
        self.gt_coords = []

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer