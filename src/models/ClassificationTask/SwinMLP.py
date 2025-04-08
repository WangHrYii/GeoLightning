import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class SwinClassifier(LightningModule):
    def __init__(self, num_classes=10, backbone=None, optimizer=None):
        super().__init__()
        
        # Initialize the Swin Transformer backbone
        self.backbone = backbone
        self.optimizer = optimizer
        
        # Get the output feature dimension from the backbone
        feature_dim = self.backbone.num_features[-1]
        
        # Classification head
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        # Extract features from the backbone
        features = self.backbone(x)
        
        # Use the output from the last stage
        x = features[-1]
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        
        # Log metrics
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        
        return loss
    
    def configure_optimizers(self):
        # Define optimizer
        optimizer = self.optimizer
        
        # Define learning rate scheduler
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs
            ),
            'interval': 'epoch',
            'name': 'lr'
        }
        
        return [optimizer], [scheduler]

