from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from codes.dataset.midog2021_dataset import MIDOG2021Dataset, midog_collate_fn
from codes.framework.framework import SegmentationModel


# Create datasets
DATA_ROOT_PATH = "/lunit/data/midog_2021_patches"
trn_dataset = MIDOG2021Dataset(
    root_path=DATA_ROOT_PATH, 
    scanners=["Aperio_CS2", "Hamamatsu_S360"], 
    training=True, 
    do_fda=True,
)
val_dataset = MIDOG2021Dataset(
    root_path=DATA_ROOT_PATH, 
    scanners=["Hamamatsu_XR"], 
    training=False, 
    do_fda=False,
)

# Create dataloader
bs = 16
trn_loader = DataLoader(trn_dataset, batch_size=bs, shuffle=True, collate_fn=midog_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, collate_fn=midog_collate_fn)

# Instantiate model, loss, and lightning module
model = YourPredefinedSegmentationModel()  # TODO
loss = torch.nn.CrossEntropyLoss()  # TODO
lightning_model = SegmentationModel(model, loss)

# Define the trainer and start training
trainer = Trainer(
    max_epochs=10, 
    check_val_every_n_epoch=5,
    gpus=1,
)
trainer.fit(lightning_model, trn_loader, val_loader)

# Evaluation
trainer.validate(lightning_model, val_loader)