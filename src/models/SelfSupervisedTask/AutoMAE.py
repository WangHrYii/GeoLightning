# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import PatchEmbed
import timm.models.vision_transformer as timm_vt
from models.blocks.pos_embed import get_2d_sincos_pos_embed
from torch.nn.utils.spectral_norm import spectral_norm

# Lightning imports
from lightning import LightningModule
from typing import Any, Dict, Optional, Tuple, Union, List, Callable
import random
import math

class Attention(timm_vt.Attention):
    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if return_attention:
            return x, attn
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_norm=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = timm_vt.DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = timm_vt.Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x), return_attention=True)
        if return_attention:
            return attn
        y = self.drop_path(y)
        x = x + y
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ScorerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # Block(768, 1) # nn.Sequential(Block(768, 1), nn.Linear(768, 1))
        self.main = nn.Sequential(nn.Conv2d(12, 6, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(6, 1, kernel_size=3, padding=1), nn.Flatten(), nn.LogSoftmax(-1))
        # self.prob_fc = nn.Sequential(nn.Linear(12, 6), nn.ReLU(inplace=True), nn.Linear(6, 2), nn.Sigmoid())
    def forward(self, x):
        # fg = F.log_softmax(self.main(x).permute(0, 2, 1), dim=-1)
        # prob = x.reshape(x.shape[0], x.shape[1], -1)
        # prob = torch.mean(prob, dim=2)
        # prob = self.prob_fc(prob)[:, None, :]
        return self.main(x) #fg, prob

class Wasserstein(nn.Module):
    def forward(self, pred_real, pred_fake=None):
        if pred_fake is not None:
            loss_real = -pred_real.mean()
            #loss_fake = pred_fake.mean()
            loss = loss_real #+ loss_fake
            return loss
        else:
            loss = -pred_real.mean()
            return loss

class Discriminator(nn.Module):
    def __init__(self, use_spectral_norm = True, use_sigmoid = True):
        super().__init__()
        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, bias=not use_spectral_norm)),
            nn.Sigmoid() if use_sigmoid else nn.Identity(),
            nn.Flatten()
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        output = conv3
        return output

class AdversarialLoss(nn.Module):
    """
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='nsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        type = nsgan | lsgan | hinge
        """
        super(AdversarialLoss, self).__init__()

        self.loss_type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

    def forward(self, outputs, is_real, is_disc=None):
        if self.loss_type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)
            return loss

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, scorer=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.grid_size = img_size // patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_norm=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        if scorer:
            # self.scorer_norm = norm_layer(12)
            self.scorer_pred = ScorerBlock()
        else:
            self.scorer_pred = None
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio, mask=None, mask_factor=0.0):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        if mask is not None:
            mask = mask.reshape(N, L)
            indices = torch.topk(mask, k=len_keep)[1]
            bindices = torch.arange(N, device=indices.device).repeat_interleave(len_keep).reshape(N, len_keep)
            bmask = torch.zeros_like(mask, device=mask.device)
            bmask[bindices, indices] = 1
            mask = bmask - mask.detach() + mask
            noise = torch.rand(N, L, device=x.device) + mask * mask_factor
            sc = noise.unsqueeze(2)
            x = sc * x + (1 - sc).detach() * x
        else:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, mask=None, mask_factor=0.0, return_ids_keep=False):
        # embed patches
        if mask is not None:
            # N = x.shape[0]  # batch, length, dim
            # L = 196
            # len_keep = int(L * (1 - mask_ratio))
            # assert len(mask.shape) == 3, mask.shape
            # mask = self.scorer_mask(mask)
            # mask_prob = mask
            assert len(mask.shape) == 3, mask.shape
            mask = self.scorer_mask(mask)
            mask_prob = mask
            #mask = F.log_softmax(mask, dim=-1)
            #
        else:
            mask_prob = None
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio, mask, mask_factor)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if mask_prob is not None:
            mask = (mask, mask_prob)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def get_last_selfattention(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                return blk(x, return_attention=True)

    def scorer_mask(self, mask):
        mask = mask.reshape(mask.shape[0], mask.shape[1], self.grid_size, self.grid_size)
        mask = self.scorer_pred(mask)
        before_mask = mask
        mask = F.gumbel_softmax(mask, tau=1) # torch.sigmoid(mask)  torch.sigmoid(mask)
        # mask = mask.mean(1)
        return mask#, before_mask

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75, mask=None, mask_factor=0.0, mode='normal'):
        if mode == 'mask':
            return self.scorer_mask(imgs)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, mask, mask_factor)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask if not isinstance(mask, tuple) else mask[0])
        return loss, pred, mask


def random_rectangle_simple(mask_factor: float, grid_size: int) -> torch.Tensor:
    """Generate a random rectangular mask for training."""
    area = random.randint(40, 160)
    l = random.randint(4, grid_size - 4)
    w = min(grid_size, math.ceil(area / l))
    x = random.randint(0, grid_size - l)
    y = random.randint(0, grid_size - w)
    mask = torch.zeros(grid_size, grid_size)
    mask[y:y+w, x:x+l] = mask_factor 
    mask += torch.rand_like(mask)
    return mask.flatten()


class AutoMAELM(LightningModule):
    """AutoMAE Lightning Module with adversarial training."""
    
    def __init__(
        self,
        # Model parameters
        model_name: str = 'mae_vit_base_patch16',
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        norm_pix_loss: bool = False,
        
        # Training parameters
        mask_ratio: float = 0.75,
        mask_factor: float = 0.5,
        loss_g_factor: float = 0.2,
        
        # Optimizer parameters
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
        betas: Tuple[float, float] = (0.9, 0.95),
        warmup_epochs: int = 40,
        min_lr: float = 0.0,
        
        # Pretrained weights
        pretrained_weight: Optional[str] = None,
        
        # Discriminator parameters
        use_spectral_norm: bool = True,
        use_sigmoid: bool = False,
        adversarial_loss_type: str = 'lsgan',
    ):
        """
        Initialize AutoMAE Lightning Module.
        
        Args:
            model_name: Name of the MAE model architecture
            img_size: Input image size
            patch_size: Vision transformer patch size
            in_chans: Number of input channels
            embed_dim: Embedding dimension
            depth: Number of transformer layers
            num_heads: Number of attention heads
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder layers
            decoder_num_heads: Number of decoder attention heads
            mlp_ratio: MLP expansion ratio
            norm_pix_loss: Whether to use normalized pixel loss
            mask_ratio: Masking ratio for MAE
            mask_factor: Factor for mask generation
            loss_g_factor: Weight for generator loss
            learning_rate: Learning rate
            weight_decay: Weight decay
            betas: Adam optimizer betas
            warmup_epochs: Number of warmup epochs
            min_lr: Minimum learning rate
            pretrained_weight: Path to pretrained scorer weights
            use_spectral_norm: Whether to use spectral normalization in discriminator
            use_sigmoid: Whether to use sigmoid in discriminator
            adversarial_loss_type: Type of adversarial loss ('lsgan', 'nsgan', 'hinge')
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize models
        self.model = MaskedAutoencoderViT(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            norm_pix_loss=norm_pix_loss,
            scorer=True
        )
        
        self.discriminator = Discriminator(
            use_spectral_norm=use_spectral_norm,
            use_sigmoid=use_sigmoid
        )
        
        # Initialize scorer (pretrained model)
        if pretrained_weight:
            self.scorer = MaskedAutoencoderViT(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                decoder_embed_dim=decoder_embed_dim,
                decoder_depth=decoder_depth,
                decoder_num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                norm_pix_loss=norm_pix_loss
            )
            # Load pretrained weights if provided
            try:
                checkpoint = torch.load(pretrained_weight, map_location="cpu")
                self.scorer.load_state_dict(checkpoint["model"])
                self.scorer.requires_grad_(False)
                print(f"Loaded pretrained scorer from {pretrained_weight}")
            except FileNotFoundError:
                print(f"Warning: Pretrained weight file {pretrained_weight} not found. Using random initialization.")
                self.scorer = None
        else:
            self.scorer = None
        
        # Loss functions
        self.criterion = AdversarialLoss(type=adversarial_loss_type)
        
        # Store hyperparameters
        self.mask_ratio = mask_ratio
        self.mask_factor = mask_factor
        self.loss_g_factor = loss_g_factor
        
        # Calculate grid size
        self.grid_size = img_size // patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Automatic optimization off for GAN training
        self.automatic_optimization = False

    def configure_optimizers(self):
        """Configure optimizers for generator and discriminator."""
        # Generator optimizer
        optimizer_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay
        )
        
        # Discriminator optimizer  
        optimizer_d = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas
        )
        
        return [optimizer_g, optimizer_d]

    def training_step(self, batch, batch_idx):
        """Training step with adversarial training."""
        optimizer_g, optimizer_d = self.optimizers()
        
        # Parse batch
        if len(batch) == 3:
            samples, _, masks = batch
            masks = masks.to(self.device, non_blocking=True)
        else:
            samples = batch
            masks = None
            
        samples = samples.to(self.device, non_blocking=True)
        batch_size = samples.shape[0]
        
        # Generate random masks for discriminator training
        with torch.no_grad():
            mask_tr = torch.zeros(batch_size, self.num_patches)
            for i in range(batch_size):
                mask_true = random_rectangle_simple(0.5, self.grid_size)
                mask_tr[i] = mask_true
            mask_tr = mask_tr.to(self.device, non_blocking=True)
            
            # Get attention maps from scorer if available
            if self.scorer is not None:
                attn_map = self.scorer.get_last_selfattention(samples)[:, :, 0, 1:]
            else:
                attn_map = torch.randn(batch_size, 12, self.num_patches).to(self.device)
                
            # Generate mask probabilities
            mask_prob_d = self.model(attn_map, mode='mask')
            mask_prob_d = mask_prob_d.reshape(-1, 1, self.grid_size, self.grid_size)
            mask_tr = F.softmax(mask_tr, dim=-1)
            mask_tr = mask_tr.reshape(-1, 1, self.grid_size, self.grid_size)

        # Train Discriminator
        optimizer_d.zero_grad()
        
        loss_d1r = self.criterion(self.discriminator(mask_tr), is_real=True)
        loss_d1f = self.criterion(self.discriminator(mask_prob_d), is_real=False)
        loss_d = loss_d1r + loss_d1f
        
        self.manual_backward(loss_d)
        optimizer_d.step()
        
        # Freeze discriminator for generator training
        for p in self.discriminator.parameters():
            p.requires_grad_(False)

        # Train Generator (MAE model)
        optimizer_g.zero_grad()
        
        # Forward pass through MAE
        loss_mae, pred, (masks_actual, mask_prob) = self.model(
            samples, 
            mask_ratio=self.mask_ratio, 
            mask=attn_map, 
            mask_factor=self.mask_factor
        )
        
        # Generator adversarial loss
        loss_g = self.criterion(
            self.discriminator(mask_prob.reshape(-1, 1, self.grid_size, self.grid_size)), 
            is_real=True
        )
        
        # Total generator loss
        total_loss = loss_mae + self.loss_g_factor * loss_g
        
        self.manual_backward(total_loss)
        optimizer_g.step()
        
        # Unfreeze discriminator
        for p in self.discriminator.parameters():
            p.requires_grad_(True)

        # Log metrics
        self.log('train_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_mae_loss', loss_mae, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_g_loss', loss_g, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_d_loss', loss_d, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_d_real', loss_d1r, on_step=True, on_epoch=True)
        self.log('train_d_fake', loss_d1f, on_step=True, on_epoch=True)
        
        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if len(batch) == 3:
            samples, _, masks = batch
        else:
            samples = batch
            
        samples = samples.to(self.device, non_blocking=True)
        
        # Forward pass
        with torch.no_grad():
            if self.scorer is not None:
                attn_map = self.scorer.get_last_selfattention(samples)[:, :, 0, 1:]
            else:
                attn_map = torch.randn(samples.shape[0], 12, self.num_patches).to(self.device)
                
            loss_mae, pred, mask_info = self.model(
                samples, 
                mask_ratio=self.mask_ratio, 
                mask=attn_map, 
                mask_factor=self.mask_factor
            )
        
        self.log('val_loss', loss_mae, prog_bar=True, on_epoch=True)
        return loss_mae

    def forward(self, x):
        """Forward pass for inference."""
        return self.model(x)


def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# # set recommended archs
# mae_vit_small_patch16= mae_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
# mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks