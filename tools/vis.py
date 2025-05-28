import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from typing import List, Dict

def visualize_batch_results(
    original_imgs: torch.Tensor, 
    sam_masks: List[Dict],
    hr_feats: torch.Tensor,
    f_mask_bicubic: torch.Tensor,
    title: str = "Batch Results",
    max_images: int = 8  # 限制显示的最大图像数
) -> List[Image.Image]:
    """
    对一个批次的结果进行可视化，将原图、掩码、高分辨率特征和掩码优化特征并排展示
    
    参数:
        original_imgs: 原始图像 [B, C, H, W]
        sam_masks: SAM掩码信息列表，格式为 [batch_masks_info1, batch_masks_info2, ...]
        hr_feats: 高分辨率特征张量 [B, C, H, W]
        f_mask_bicubic: 掩码优化后的特征张量 [B, C, H, W]
        title: 可视化标题
        max_images: 最多显示的图像数量
        
    返回:
        combined_images: 合并后的图像列表，每个元素对应一个批次样本的可视化结果
    """
    if original_imgs.ndim == 5:
        original_imgs = original_imgs.squeeze(0)
        hr_feats = hr_feats.squeeze(0)
        f_mask_bicubic = f_mask_bicubic.squeeze(0)
        sam_masks = sam_masks[0]
    batch_size = min(original_imgs.shape[0], max_images)
    combined_images = []
    
    # 获取特征可视化
    hr_feats_vis = visualize_features(hr_feats)
    f_mask_bicubic_vis = visualize_features(f_mask_bicubic)
    
    # 为批次中的每个样本创建可视化
    for i in range(batch_size):
        # 1. 获取原始图像
        orig_img = original_imgs[i]
        
        # 2. 提取SAM掩码
        mask_img = None
        if i < len(sam_masks) and sam_masks[i]:
            # 创建掩码可视化
            mask_vis = create_mask_visualization(sam_masks[i], orig_img)
            mask_img = torch.tensor(np.array(mask_vis)).permute(2, 0, 1) / 255.0
        else:
            # 创建空白掩码
            mask_img = torch.zeros_like(orig_img)
        
        # 获取特征可视化
        hr_feat = hr_feats_vis[i]
        f_mask = f_mask_bicubic_vis[i]
        
        # 创建单个样本的合并图像
        combined = combine_four_images(
            orig_img, 
            mask_img, 
            hr_feat, 
            f_mask, 
            titles=["Original Image", "SAM Mask", "High-Res Features", "Mask-Optimized Features"],
            main_title=f"{title} - Sample #{i+1}"
        )
        
        combined_images.append(combined)
    
    return combined_images

def combine_four_images(img1, img2, img3, img4, titles=None, main_title=None):
    """
    将四个图像合并为2×2网格的单个图像
    
    参数:
        img1-img4: 要合并的四个图像（可以是张量或PIL图像）
        titles: 四个子图的标题列表
        main_title: 整个图像的主标题
        
    返回:
        combined_image: 合并后的PIL图像
    """
    # 转换为PIL图像
    if isinstance(img1, torch.Tensor):
        img1 = T.ToPILImage()(img1.cpu())
    if isinstance(img2, torch.Tensor):
        img2 = T.ToPILImage()(img2.cpu())
    if isinstance(img3, torch.Tensor):
        img3 = T.ToPILImage()(img3.cpu())
    if isinstance(img4, torch.Tensor):
        img4 = T.ToPILImage()(img4.cpu())
    
    # 确保所有图像大小相同
    width, height = img2.size
    img1 = img1.resize((width, height))
    img3 = img3.resize((width, height))
    img4 = img4.resize((width, height))
    
    # 创建1×4网格
    margin = 10  # 边距
    title_height = 30 if titles else 0
    main_title_height = 40 if main_title else 0
    
    # 计算总宽度和高度（包含边距和标题）
    total_width = width * 4 + margin * 5
    total_height = height + margin * 2 + title_height + main_title_height
    
    # 创建白色背景图像
    combined_image = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    
    # 粘贴四个图像到一行
    combined_image.paste(img1, (margin, margin + main_title_height + title_height))
    combined_image.paste(img2, (margin * 2 + width, margin + main_title_height + title_height))
    combined_image.paste(img3, (margin * 3 + width * 2, margin + main_title_height + title_height))
    combined_image.paste(img4, (margin * 4 + width * 3, margin + main_title_height + title_height))
    
    # 添加标题
    draw = ImageDraw.Draw(combined_image)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
        title_font = ImageFont.truetype("Arial.ttf", 18) if main_title else None
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # 添加标题
    draw = ImageDraw.Draw(combined_image)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
        title_font = ImageFont.truetype("Arial.ttf", 18) if main_title else None
    except IOError:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # 添加子标题，放在每个图像上方
    if titles:
        draw.text((margin + width//2 - 40, margin + main_title_height), titles[0], fill=(0, 0, 0), font=font)
        draw.text((margin * 2 + width + width//2 - 40, margin + main_title_height), titles[1], fill=(0, 0, 0), font=font)
        draw.text((margin * 3 + width * 2 + width//2 - 40, margin + main_title_height), titles[2], fill=(0, 0, 0), font=font)
        draw.text((margin * 4 + width * 3 + width//2 - 40, margin + main_title_height), titles[3], fill=(0, 0, 0), font=font)
    
    # 添加主标题，放在最上方居中
    if main_title and title_font:
        draw.text((total_width // 2 - 100, margin), main_title, fill=(0, 0, 0), font=title_font)
    
    return combined_image



def visualize_features(
    features: torch.Tensor
) -> torch.Tensor:
    """
    可视化特征图
    
    参数:
        features: 特征图张量 [B, C, H, W]
    返回:
        vis_images: 可视化后的图像
    """
    # 确保特征在CPU上
    features = features.detach().cpu()
    batch_size = features.shape[0]
    
    # 创建批次网格
    vis_images = []
    for i in range(batch_size):
        # 将特征归一化到 [0, 1] 范围
        C, H, W = features[i].shape
        features_norm = features[i].clone()
        
        # 对每个通道分别归一化
        for c in range(C):
            channel = features[i, c]
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:
                features_norm[c] = (channel - min_val) / (max_val - min_val)
        
        # 如果通道数大于3，使用PCA降维
        if C > 3:
            features_flat = features_norm.view(C, -1).transpose(0, 1)
            U, S, V = torch.pca_lowrank(features_flat, q=3)
            pca_features = torch.matmul(features_flat, V[:, :3]).view(H, W, 3)
            pca_features = pca_features.permute(2, 0, 1)  # [3, H, W]
            
            # 归一化PCA结果
            for c in range(3):
                channel = pca_features[c]
                min_val = channel.min()
                max_val = channel.max()
                if max_val > min_val:
                    pca_features[c] = (channel - min_val) / (max_val - min_val)
            
            result = pca_features
        else:
            # 如果通道数小于等于3，直接使用前3个通道或填充
            result = torch.zeros(3, H, W, device=features.device)
            for c in range(min(C, 3)):
                result[c] = features_norm[c]
        
        # 转换为PIL图像
        result = result.numpy().transpose(1, 2, 0)
        result = (result * 255).astype(np.uint8)
        result = Image.fromarray(result)        
        vis_images.append(result)
    
    return vis_images


def create_mask_visualization(masks_info, original_image=None):
    """
    创建掩码可视化
    
    参数:
        masks_info: 掩码信息列表
        original_image: 原始图像
        
    返回:
        mask_vis: PIL图像
    """
    if len(masks_info) == 0:
        return
    height, width = masks_info[0]['mask'].shape
    img = np.ones((height, width, 4), dtype=np.uint8) * 255  # 白底
    img[:, :, 3] = 0

    alpha = int(0.6 * 255)
    sorted_anns = sorted(masks_info, key=lambda x: x['area'], reverse=False)
    for ann in sorted_anns:
        m = ann['mask'].cpu().numpy().astype(np.bool_)  # (H, W)
        color = np.random.randint(0, 255, 3, dtype=np.uint8)
        color_mask = np.zeros((height, width, 4), dtype=np.uint8)
        color_mask[:, :, :3] = color
        color_mask[:, :, 3] = alpha
        img[m] = color_mask[m]

    if original_image is not None:
        image_np = original_image.cpu().permute(1, 2, 0).numpy()
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        image_pil = Image.fromarray(image_np)
        image_pil = image_pil.resize((width, height), resample=Image.BILINEAR)
        base_img = np.array(image_pil)
    else:
        base_img = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Alpha blending
    img_float = img.astype(np.float32)
    base_img_float = base_img.astype(np.float32)
    alpha_mask = img_float[:, :, 3:4] / 255.0

    overlay_img = (1 - alpha_mask) * base_img_float + alpha_mask * img_float[:, :, :3]
    overlay_img = np.clip(overlay_img, 0, 255).astype(np.uint8)

    image_np = original_image.cpu().permute(1, 2, 0).numpy()
    if image_np.max() <= 1.0:
        image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    # 转换为PIL图像然后调整大小
    image_pil = Image.fromarray(image_np)
    image_pil = image_pil.resize((img.shape[1], img.shape[0]))
    
    # 转回NumPy数组以进行操作
    overlay_img = np.array(image_pil)
    for i in range(3):
        overlay_img[:, :, i] = overlay_img[:, :, i] * (1 - img[:, :, 3]/255) + img[:, :, i] * (img[:, :, 3]/255)
   
    # 转换为PIL图像
    return Image.fromarray(overlay_img)

