import logging
import os
from dataclasses import asdict

import wandb
import numpy as np
import torch
from lightly.loss import NTXentLoss
from torch.optim import Optimizer

from src.configuration import Config
from src.constants import LATEST_MODEL_FILE_NAME
from src.model.simclr import SimCLR


class Trainer:
    def __init__(self, model: SimCLR, optimiser: Optimizer, lr_schedule, config: Config):
        self.model = model
        self.optimiser = optimiser
        self.lr_schedule = lr_schedule
        self.config = config

        self.criterion = NTXentLoss()

        if config.general.log_to_wandb:
            wandb.init(
                project='SimCLR david', 
                config={
                    
                    "epochs": 100,
                    "learning rate": 0.1,
                    "image_size": 512,
                    "crop": "with no crop",
                    "batch_size": 32,
                    "warm up": 5
                    },
                name='SIMCLR v4',
                
            )

    def train_one_epoch(self, train_loader, epoch):
        train_loss = 0.0

        for batch_num, batch in enumerate(train_loader):

            global_iteration = len(train_loader) * epoch + batch_num

            self.optimiser.param_groups[0]['lr'] = self.lr_schedule[global_iteration]

            # view1 = batch[0].to(self.config.optim.device)
            # view2 = batch[1].to(self.config.optim.device)
            
            
            # print(f"Type of batch: {type(batch)}")
            # print(f"Length of batch: {len(batch)}")
            # print(f"Type of batch[0]: {type(batch[0])}")
            # print(f"Length of batch[0]: {len(batch[0])}")
            # if len(batch[0]) > 0:
            #     print(f"Type of batch[0][0]: {type(batch[0][0])}")
            #     if isinstance(batch[0][0], torch.Tensor):
            #         print(f"Shape of batch[0][0]: {batch[0][0].shape}")

            view1 = torch.stack([item.to(self.config.optim.device) for item in batch[0]])
            view2 = torch.stack([item.to(self.config.optim.device) for item in batch[1]])

            # # More debug prints
            # # print(f"Shape of view1 after stacking: {view1.shape}")
            # # print(f"Shape of view2 after stacking: {view2.shape}")

            # # Reshape if necessary
            view1 = view1.view(-1, 3, 512, 512)
            view2 = view2.view(-1, 3, 512, 512)

            # print(f"Final shape of view1: {view1.shape}")
            # print(f"Final shape of view2: {view2.shape}")

            z1 = self.model(view1)
            z2 = self.model(view2)

            loss = self.criterion(z1, z2)

            train_loss += float(loss.item())

            loss.backward()
            self.optimiser.step()
            self.optimiser.zero_grad()

        train_loss /= len(train_loader)

        return {
            'train/loss': train_loss,
            'learning rate': self.optimiser.param_groups[0]['lr']
        }

    def validate_one_epoch(self, val_loader):
        self.model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                # view1 = batch[0].to(self.config.optim.device)
                # view2 = batch[1].to(self.config.optim.device)
                view1 = torch.stack([item.to(self.config.optim.device) for item in batch[0]])
                view2 = torch.stack([item.to(self.config.optim.device) for item in batch[1]])


                # Reshape if necessary
                view1 = view1.view(-1, 3, 512, 512)
                view2 = view2.view(-1, 3, 512, 512)
                
                z1 = self.model(view1)
                z2 = self.model(view2)

                loss = self.criterion(z1, z2)

                val_loss += float(loss.item())

        self.model.train()

        val_loss /= len(val_loader)

        return {
            'val/loss': val_loss,
        }

    def train(self, train_loader, val_loader, start_epoch):
        best_val_loss = np.inf
        
        for epoch in range(start_epoch, self.config.optim.epochs):
            logging.info('Epoch %s/%s', epoch + 1, self.config.optim.epochs)

            train_metrics = self.train_one_epoch(train_loader, epoch)

            val_metrics = self.validate_one_epoch(val_loader)

            if self.config.general.log_to_wandb:
                wandb.log({**train_metrics, **val_metrics})

            logging.info({**train_metrics, **val_metrics})

            val_loss = val_metrics['val/loss']

            state_dict = {
                'model': self.model.backbone.state_dict(), #added backbone
                'optimiser': self.optimiser.state_dict(),
                'epoch': epoch + 1,
            }

            torch.save(
                state_dict,
                os.path.join(self.config.general.output_dir, LATEST_MODEL_FILE_NAME)
            )

            if epoch % self.config.general.checkpoint_freq == 0 or (epoch + 1) == self.config.optim.epochs:
                torch.save(
                    state_dict,
                    os.path.join(self.config.general.output_dir, f'simclr_medical-epoch-{epoch}.pth')
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    state_dict,
                    os.path.join(self.config.general.output_dir, 'best.pth')
                )
