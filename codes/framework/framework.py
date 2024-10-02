import torch
import pytorch_lightning as pl
import torch.nn.functional as F

class SegmentationModel(pl.LightningModule):
    def __init__(self, model, loss, lr):
        super(SegmentationModel, self).__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, masks = batch
        preds = self.forward(images)
        loss = self.loss(preds, masks)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        preds = self.forward(images)
        val_loss = self.loss(preds, masks)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer