"""
Implements training routine.
"""

from pathlib import Path
import torch
import pytorch_lightning as pl
from models.ddpm import create_ddpm
from schemas import TrainInput, TrainOutput


#1. Defines useful functions.
def train_model(input: TrainInput) -> TrainOutput:
    """
    1. Loads train and validation data loaders.
    2. Retrieves specified model: ddpm.
    3. Creates trainer instance. 
    4. Performs training routine.
    5. Saves model on the specified folder. 
    """

    #1. Retrieves training and test dataloaders.
    train_loader_path, test_loader_path = input.split_train_test
    train_loader, test_loader = torch.load(train_loader_path, weights_only= False), torch.load(test_loader_path, weights_only= False)

    #2. Retrieves specified model. 
    model = create_ddpm(timesteps= input.timesteps,
                        image_size= input.image_size[0],
                        in_channel= input.in_channel, 
                        base_dim= input.base_dim,
                        dim_mults= input.dim_mults,
                        total_steps_factor= input.total_steps_factor)

    #3. Creates trainer.
    trainer = pl.Trainer(max_epochs = input.max_epochs, callbacks= None) #. Fix callback later.

    #4. Initializes training routine. 
    trainer.fit(model = model, train_dataloaders = train_loader, val_dataloaders= test_loader)

    #5. Saves model. 
    parent_path: Path = train_loader_path.parent.parent.parent.parent
    model_dir = parent_path / "trained_models" 
    model_dir.mkdir(exist_ok= True, parents= True)

    model_name = "{}.ckpt".format(input.model_name)
    model_path = model_dir / model_name
    trainer.save_checkpoint(model_path)

    return TrainOutput(model_checkpoint_path= model_path)
    