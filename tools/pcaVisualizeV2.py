import os
import os.path as osp

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

import torch
import torchvision.transforms as T

def main(image_file, save_fg_mask=False, img_size=224, output_folder="outputs"):
    os.makedirs(output_folder, exist_ok=True)

    assert img_size % 14 == 0, "The image size must be exactly divisible by 14"

    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_vits14 = dinov2_vits14.cuda()
    dinov2_vits14.eval()

    transform = T.Compose([
                    T.ToTensor(), 
                    T.Resize(img_size+int(img_size*0.01)*10), 
                    T.CenterCrop(img_size), 
                    T.Normalize([0.5], [0.5]), 
                ])
    
    patch_h = patch_w = img_size // 14
    
    # Load the image and select the first three bands
    image = Image.open(image_file)
    image = np.array(image)[:, :, :3]  # Select the first three bands
    image = Image.fromarray(image)
    
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    images = torch.FloatTensor(image).cuda()
    images_plot = ((images.cpu().numpy()*0.5+0.5)*255).transpose(0, 2, 3, 1).astype(np.uint8)

    with torch.no_grad():
        embeddings = dinov2_vits14.forward_features(images)
        x_norm_patchtokens = embeddings["x_norm_patchtokens"].cpu().numpy()

    x_norm_1616_patches = x_norm_patchtokens.reshape(-1, x_norm_patchtokens.shape[-1])

    fg_pca = PCA(n_components=1)  # n_components=1表示降维到1维
    fg_pca_images = fg_pca.fit_transform(x_norm_1616_patches)
    fg_pca_images = minmax_scale(fg_pca_images)  # 将数据缩放到[0, 1]区间
    fg_pca_images = fg_pca_images.reshape(1, patch_h * patch_w)

    masks = []
    for i in range(1):
        image_patches = fg_pca_images[i,:]
        mask = (image_patches > 0.1).ravel()  # 0.4是一个经验值，可以根据实际情况调整，作用是将背景和前景分开
        masks.append(mask)

    if save_fg_mask:
        for i in range(1):
            mask = masks[i]
            image_patches = fg_pca_images[i,:]
            image_patches[np.logical_not(mask)] = 0
            image_patches = image_patches.reshape([patch_h, patch_w])

            plt.subplot(221+i)
            plt.imshow(images_plot[i])
            plt.imshow(image_patches, extent=(0,img_size,img_size,0), alpha=0.5)
        plt.savefig(osp.join(output_folder, "fg_mask.jpg"))
        plt.close()

    pca = PCA(n_components=3)
    fg_patches = x_norm_1616_patches[masks[0], :]
    pca_features = pca.fit_transform(fg_patches)
    fg_result = minmax_scale(pca_features)

    mask_indices = [0, np.sum(masks[0]), -1]

    plt.figure(figsize=(8, 8))
    for i in range(1):
        plt.subplot(1, 2, 2*i+1)
        plt.axis("off")
        plt.imshow(images_plot[i])

        plt.subplot(1, 2, 2*i+2)
        plt.axis("off")
        pca_results = np.zeros((patch_h*patch_w, 3), dtype='float32')
        pca_results[masks[i]] = fg_result[mask_indices[i]:mask_indices[i+1]]
        pca_results = pca_results.reshape(patch_h, patch_w, 3)
        plt.imshow(pca_results)
    plt.savefig(osp.join(output_folder, "results.jpg"))
    plt.close()

    ### Raw
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(x_norm_1616_patches)
    pca_features = pca_features.reshape(1, patch_h*patch_w, 3)
    for i in range(1):
        mask = masks[i]
        pca_features[i, ~mask] = np.min(pca_features[i])
    pca_features = pca_features.reshape(patch_h*patch_w, -1)
    fg_result = minmax_scale(pca_features)
    fg_result = fg_result.reshape(1, patch_h, patch_w, 3)

    plt.figure(figsize=(8, 8))
    for i in range(1):
        plt.subplot(1, 2, 2*i+1)
        plt.axis("off")
        plt.imshow(images_plot[i])

        plt.subplot(1, 2, 2*i+2)
        plt.axis("off")
        plt.imshow(fg_result[i])
    plt.savefig(osp.join(output_folder, "results_raw.jpg"))
    plt.close()


if __name__ == "__main__":
    image_file = "images/3.jpg"
    save_fg_mask = True
    img_size = 448
    output_folder = "outputs/3_01/"
    main(image_file, save_fg_mask, img_size, output_folder)