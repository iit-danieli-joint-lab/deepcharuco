import torch
from torch.utils.data import DataLoader,random_split

from configs import load_configuration
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelCheckpoint
import configs
from data_refinenet import RefineDataset
from models.refinenet import RefineNet, lRefineNet
import pytorch_lightning as pl

def custom_collate(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Return None or handle this case appropriately
    return torch.utils.data.dataloader.default_collate(batch)

if __name__ == '__main__':
    config = load_configuration(configs.CONFIG_PATH)
    total = 4
    print('before')
    # Load the entire dataset (train + val will come from here)
    full_dataset = RefineDataset(config, config.train_labels, config.train_images,
                                 visualize=False, validation=False, total=total)
    print('after')
    # Split the dataset into train and validation sets (e.g., 80% train, 20% validation)
    train_size = int(0.8 * len(full_dataset))  # 80% training
    val_size = len(full_dataset) - train_size  # 20% validation
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(train_size)
    print(val_size)
    print(len(train_dataset))
    print(len(val_dataset))
    
    # Set batch sizes (distributed across devices if needed)
    bs_train = max(1, config.bs_train_rn // total)
    bs_val = max(1, config.bs_val_rn // total)
    
    print(f'Using virtual train batch size of {bs_train} images')
    print(f'Using virtual val batch size of {bs_val} images')

    # Create DataLoader for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=bs_train,
                              shuffle=True, num_workers=config.num_workers,
                              collate_fn=custom_collate,
                              pin_memory=True, prefetch_factor=10)
    
    val_loader = DataLoader(val_dataset, batch_size=bs_val,
                            shuffle=False, num_workers=config.num_workers,
                            collate_fn=custom_collate,
                            pin_memory=True, prefetch_factor=10)

    # Initialize the model
    model = RefineNet()
    train_model = lRefineNet(model)

    # Set up TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="refinenet")

    # No ModelCheckpoint (as per your request)
    checkpoint_callback = ModelCheckpoint(dirpath="tb_logs/ckpts_refinenet/", save_top_k=5, monitor="val_refinenet_loss")

    # Trainer setup
    trainer = pl.Trainer(
        max_epochs=100,
        gpus="auto",  # Adjust according to your hardware (e.g., 0 for CPU or list of GPU indices)
        logger=logger,
        # No checkpointing callbacks
        callbacks=[checkpoint_callback]  # Optional, if you want SWA
    )

    # Train the model
    trainer.fit(train_model, train_loader, val_loader)    
   
