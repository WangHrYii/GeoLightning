import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片（替换为你的图片路径）
img_path = '/home/whr/Codes/CLFoundation/CLFoundation/images/3.jpg'
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

# 初始化SLIC超像素分割器
region_size = 100         # 超像素区域大小（像素）
ruler = 0                # 紧凑度因子（值越大形状越规则）
num_iterations = 1000    # 迭代次数

slic = cv2.ximgproc.createSuperpixelSLIC(
    image, 
    algorithm=cv2.ximgproc.SLICO,  # 使用SLICO算法（改进的SLIC）
    region_size=region_size,
    ruler=ruler
)

# 执行超像素分割
slic.iterate(num_iterations)

# 获取结果
labels = slic.getLabels()          # 获取每个像素的标签
mask = slic.getLabelContourMask()  # 获取超像素边界mask

# 在原图上绘制边界（绿色）
contour_color = [0, 255, 0]
image_contour = image.copy()
image_contour[mask == 255] = contour_color  # 将边界位置设为绿色

# 创建可视化对比图
plt.figure(figsize=(15, 8))

# 显示原图
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')

# 显示分割结果
plt.subplot(1, 2, 2)
plt.imshow(image_contour)
plt.title(f'SLIC Superpixels (Regions: {slic.getNumberOfSuperpixels()})')
plt.axis('off')

plt.tight_layout()
plt.savefig('superpixel_result.png')