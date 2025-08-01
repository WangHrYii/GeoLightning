import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from lightning import LightningModule
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import io
import swanlab
from src.losses.regression import ScaleInvariantLoss, GradientConsistencyLoss


class TreeHeightBase(LightningModule):
    """
    Base class for tree height estimation models that handles training, evaluation, and metrics.
    Models that implement this base class just need to provide their model implementation
    and configure_optimizers method.
    """
    def __init__(
        self,
        model,
        n_classes=1,
        lambda_seg=1.0,
        lambda_height=1.0,
        si_lambda=0.5,
        gradient_lambda=0.1,
    ):
        super().__init__()
        
        # Store the model
        self.model = model
        
        # Loss functions
        self.seg_loss = nn.BCEWithLogitsLoss() if n_classes == 1 else nn.CrossEntropyLoss()
        self.height_si_loss = ScaleInvariantLoss(lambd=si_lambda)
        self.height_grad_loss = GradientConsistencyLoss()
        
        # Loss weights
        self.lambda_seg = lambda_seg
        self.lambda_height = lambda_height
        self.gradient_lambda = gradient_lambda
        
        # Metrics
        self.train_iou = torchmetrics.JaccardIndex(task="binary" if n_classes == 1 else "multiclass", num_classes=n_classes)  # binary表示二分类，此时iou计算的是两个类别的交集除以并集
        self.val_iou = torchmetrics.JaccardIndex(task="binary" if n_classes == 1 else "multiclass", num_classes=n_classes)
        self.test_iou = torchmetrics.JaccardIndex(task="binary" if n_classes == 1 else "multiclass", num_classes=n_classes)
        
        # F1 metrics
        self.train_f1 = torchmetrics.F1Score(task="binary" if n_classes == 1 else "multiclass", num_classes=n_classes)
        self.val_f1 = torchmetrics.F1Score(task="binary" if n_classes == 1 else "multiclass", num_classes=n_classes)
        self.test_f1 = torchmetrics.F1Score(task="binary" if n_classes == 1 else "multiclass", num_classes=n_classes)
        
        # Regression metrics
        self.train_mae = torchmetrics.MeanAbsoluteError()
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        
        self.train_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.val_rmse = torchmetrics.MeanSquaredError(squared=False)
        self.test_rmse = torchmetrics.MeanSquaredError(squared=False)

        # R2 metrics
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()
        
        # Store n_classes for later use
        self.n_classes = n_classes
    
    def forward(self, x):
        """Forward pass through the model"""
        return self.model(x)
    
    def _compute_losses(self, batch):
        """
        Compute segmentation and height losses
        
        Args:
            batch: Input batch containing (x, seg_mask, height_map, metadata)
            
        Returns:
            total_loss: Combined loss
            seg_loss: Segmentation loss
            height_loss: Height regression loss
            seg_pred: Segmentation predictions
            height_pred: Height predictions
            seg_gt: Segmentation ground truth
            height_gt: Height ground truth
        """
        x, seg_gt, height_gt, _ = batch

        x = x / 255.0

        # Forward pass
        seg_pred, height_pred = self(x)

        seg_gt = seg_gt.unsqueeze(1)  # B, 1, H, W
        height_gt = height_gt.unsqueeze(1)  # B, 1, H, W
        
        # Compute segmentation loss - combination of BCE, Dice, and IoU losses
        if self.n_classes == 1:
            # BCE loss
            bce_loss = F.binary_cross_entropy_with_logits(seg_pred, seg_gt.float())
            
            # Dice loss
            pred_sigmoid = torch.sigmoid(seg_pred)
            smooth = 1e-5
            pred_flat = pred_sigmoid.view(-1)
            gt_flat = seg_gt.view(-1)
            intersection = (pred_flat * gt_flat).sum()
            dice_loss = 1 - (2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth)
            
            # IoU loss
            union = pred_flat.sum() + gt_flat.sum() - intersection
            iou_loss = 1 - (intersection + smooth) / (union + smooth)
            
            # Combined loss
            seg_loss = bce_loss + dice_loss + iou_loss
        else:
            seg_loss = F.cross_entropy(seg_pred, seg_gt.long())
        
        # Compute height loss - combination of L1, L2, R2 and gradient loss
        # l1_loss = F.l1_loss(height_pred, height_gt)
        l2_loss = F.mse_loss(height_pred, height_gt)
        
        # Calculate R2 loss
        mean_target = torch.mean(height_gt)
        ss_res = torch.sum((height_gt - height_pred) ** 2)
        ss_tot = torch.sum((height_gt - mean_target) ** 2)
        r2_loss = 1 - (ss_res / (ss_tot + 1e-8))

        height_grad_loss = self.height_grad_loss(height_pred, height_gt)
        
        # Combined height loss
        height_loss = l2_loss + (1 - r2_loss)*10 + height_grad_loss
        
        # Total loss
        total_loss = self.lambda_seg * seg_loss + self.lambda_height * height_loss
        
        return total_loss, seg_loss, height_loss, seg_pred, height_pred, seg_gt, height_gt
    
    def training_step(self, batch, batch_idx):
        """Base training step"""
        total_loss, seg_loss, height_loss, seg_pred, height_pred, seg_gt, height_gt = self._compute_losses(batch)
        
        # Log losses
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_seg_loss", seg_loss, on_step=False, on_epoch=True)
        self.log("train_height_loss", height_loss, on_step=False, on_epoch=True)
        
        # Log height prediction max
        self.log("train_height_max", height_pred.max(), on_step=False, on_epoch=True, reduce_fx=torch.max)
        
        # Metrics for segmentation
        if self.n_classes == 1:
            seg_pred_binary = torch.sigmoid(seg_pred) > 0.5
            self.train_iou(seg_pred_binary, seg_gt.bool())
            self.train_f1(seg_pred_binary, seg_gt.bool())
        else:
            self.train_iou(torch.argmax(seg_pred, dim=1), seg_gt)
            self.train_f1(torch.argmax(seg_pred, dim=1), seg_gt)
        
        # Metrics for height - only where trees are present
        valid_mask = seg_gt > 0
        if valid_mask.sum() > 0:
            self.train_mae(height_pred[valid_mask], height_gt[valid_mask])
            self.train_rmse(height_pred[valid_mask], height_gt[valid_mask])
            self.train_r2(height_pred[valid_mask], height_gt[valid_mask])
        
        # Log metrics
        self.log("train_iou", self.train_iou, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True)
        self.log("train_mae", self.train_mae, on_step=False, on_epoch=True)
        self.log("train_rmse", self.train_rmse, on_step=False, on_epoch=True)
        self.log("train_r2", self.train_r2, on_step=False, on_epoch=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        total_loss, seg_loss, height_loss, seg_pred, height_pred, seg_gt, height_gt = self._compute_losses(batch)
        
        # Log losses
        self.log("val_loss", total_loss, on_epoch=True, prog_bar=True)
        self.log("val_seg_loss", seg_loss, on_epoch=True)
        self.log("val_height_loss", height_loss, on_epoch=True)
        
        # Log height prediction max
        self.log("val_height_max", height_pred.max(), on_epoch=True, reduce_fx=torch.max)
        
        # Metrics for segmentation
        if self.n_classes == 1:
            seg_pred_binary = torch.sigmoid(seg_pred) > 0.5
            self.val_iou(seg_pred_binary, seg_gt.bool())
            self.val_f1(seg_pred_binary, seg_gt.bool())
        else:
            self.val_iou(torch.argmax(seg_pred, dim=1), seg_gt)
            self.val_f1(torch.argmax(seg_pred, dim=1), seg_gt)
            
        # Metrics for height - only where trees are present
        valid_mask = seg_gt > 0
        if valid_mask.sum() > 0:
            self.val_mae(height_pred[valid_mask], height_gt[valid_mask])
            self.val_rmse(height_pred[valid_mask], height_gt[valid_mask])
            self.val_r2(height_pred[valid_mask], height_gt[valid_mask])
        
        # Log metrics
        self.log("val_iou", self.val_iou, on_epoch=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        self.log("val_mae", self.val_mae, on_epoch=True, prog_bar=True)
        self.log("val_rmse", self.val_rmse, on_epoch=True)
        self.log("val_r2", self.val_r2, on_epoch=True, prog_bar=True)
        
        # Collect samples for visualization
        if batch_idx == 0:
            self.validation_samples = {
                'images': batch[0],
                'seg_gt': seg_gt,
                'height_gt': height_gt,
                'seg_pred': seg_pred,
                'height_pred': height_pred
            }
        
        return total_loss

    def on_validation_epoch_end(self):
        """Visualize validation results at the end of each epoch"""
        # Check if validation samples exist
        if not hasattr(self, 'validation_samples') or self.validation_samples is None:
            return
            
        # Randomly select up to 3 images to visualize
        batch_size = self.validation_samples['images'].shape[0]
        indices = torch.randperm(batch_size)[:min(3, batch_size)]
        
        # Create and log visualizations
        for i, idx in enumerate(indices):
            # Get sample data
            image = self.validation_samples['images'][idx]
            seg_gt = self.validation_samples['seg_gt'][idx]
            height_gt = self.validation_samples['height_gt'][idx]
            seg_pred = self.validation_samples['seg_pred'][idx]
            height_pred = self.validation_samples['height_pred'][idx]
            
            # Process segmentation prediction
            if self.n_classes == 1:
                seg_pred = torch.sigmoid(seg_pred)
            
            # Convert to numpy arrays
            image = image.cpu().numpy().transpose(1, 2, 0)
            seg_gt = seg_gt.cpu().numpy().squeeze()
            height_gt = height_gt.cpu().numpy().squeeze()
            seg_pred = seg_pred.cpu().numpy().squeeze()
            seg_pred = (seg_pred > 0.5).astype(np.uint8)
            height_pred = height_pred.cpu().numpy().squeeze()
            
            image = image.astype(np.uint8)
            
            # Create visualization with matplotlib
            fig, axes = plt.subplots(1, 5, figsize=(20, 4))
            
            # Original image
            axes[0].imshow(image)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Ground truth segmentation
            axes[1].imshow(seg_gt, cmap='gray')
            axes[1].set_title('Segmentation GT')
            axes[1].axis('off')
            
            # Ground truth height
            im3 = axes[2].imshow(height_gt, cmap='viridis')
            axes[2].set_title('Height GT')
            axes[2].axis('off')
            plt.colorbar(im3, ax=axes[2])
            
            # Predicted segmentation
            axes[3].imshow(seg_pred, cmap='gray')
            axes[3].set_title('Segmentation Pred')
            axes[3].axis('off')
            
            # Predicted height
            im5 = axes[4].imshow(height_pred, cmap='viridis')
            axes[4].set_title('Height Pred')
            axes[4].axis('off')
            plt.colorbar(im5, ax=axes[4])
            
            plt.tight_layout()
            
            # Convert to wandb format
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            pil_image = Image.open(buf)
            
            # Log to wandb
            swanlab.log({
                f"validation_sample_{i}": swanlab.Image(pil_image),
                "step": self.current_epoch
            })
            
            plt.close()
        
        # Clean up
        self.validation_samples = None

    def test_step(self, batch, batch_idx):
        """Test step"""
        total_loss, seg_loss, height_loss, seg_pred, height_pred, seg_gt, height_gt = self._compute_losses(batch)
        
        # Log losses
        self.log("test_loss", total_loss, on_epoch=True)
        self.log("test_seg_loss", seg_loss, on_epoch=True)
        self.log("test_height_loss", height_loss, on_epoch=True)
        
        # Log height prediction max
        self.log("test_height_max", height_pred.max(), on_epoch=True, reduce_fx=torch.max)
        
        # Metrics for segmentation
        if self.n_classes == 1:
            seg_pred_binary = torch.sigmoid(seg_pred) > 0.5
            self.test_iou(seg_pred_binary, seg_gt.bool())
            self.test_f1(seg_pred_binary, seg_gt.bool())
        else:
            self.test_iou(torch.argmax(seg_pred, dim=1), seg_gt)
            self.test_f1(torch.argmax(seg_pred, dim=1), seg_gt)
            
        # Metrics for height - only where trees are present
        valid_mask = seg_gt > 0
        if valid_mask.sum() > 0:
            self.test_mae(height_pred[valid_mask], height_gt[valid_mask])
            self.test_rmse(height_pred[valid_mask], height_gt[valid_mask])
            self.test_r2(height_pred[valid_mask], height_gt[valid_mask])
        
        # Log metrics
        self.log("test_iou", self.test_iou, on_epoch=True)
        self.log("test_f1", self.test_f1, on_epoch=True)
        self.log("test_mae", self.test_mae, on_epoch=True)
        self.log("test_rmse", self.test_rmse, on_epoch=True)
        self.log("test_r2", self.test_r2, on_epoch=True)
        
        return total_loss
    
    def on_train_epoch_start(self):
        """Callback for start of training epoch"""
        pass
    
    def on_train_epoch_end(self):
        """Callback for end of training epoch"""
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction step"""
        if isinstance(batch, tuple) and len(batch) >= 1:
            x = batch[0]
        else:
            x = batch
            
        seg_pred, height_pred = self(x)
        
        if self.n_classes == 1:
            seg_pred = torch.sigmoid(seg_pred)
        
        return seg_pred, height_pred
    
    def configure_optimizers(self):
        """
        Configure optimizers - should be implemented by child classes
        """
        raise NotImplementedError("Subclasses must implement configure_optimizers()") 