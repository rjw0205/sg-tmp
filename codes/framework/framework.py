import torch
import pytorch_lightning as pl
import torch.nn.functional as F

class SegmentationModel(pl.LightningModule):
    def __init__(self, model, supervised_loss, consistency_loss, lr):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.supervised_loss = supervised_loss
        self.consistency_loss = consistency_loss
        self.lr = lr
    
    def forward(self, x, fda_x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        imgs_dict, seg_labels, gt_points, gt_categories = batch
        
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
        self.log("train_supervised_loss", supervised_loss)
        self.log("train_consistency_loss", consistency_loss)

        # Return losses
        return {
            "loss": supervised_loss + consistency_loss,
            "supervised_loss": supervised_loss,
            "consistency_loss": consistency_loss,
        }

    def validation_step(self, batch, batch_idx):
        imgs_dict, seg_labels, gt_points, gt_categories = batch

        # Compute supervised loss
        imgs = imgs_dict["original"]
        preds = self.forward(imgs)
        supervised_loss = self.supervised_loss(preds, seg_labels)

        self.log("val_supervised_loss", supervised_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer