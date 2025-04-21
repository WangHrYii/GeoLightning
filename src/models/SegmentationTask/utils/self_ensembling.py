# -*- coding: utf-8 -*-
"""
utils for self-ensembling
"""
import os, copy, cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
import torch.nn.functional as F
import src.models.SegmentationTask.utils.self_ensembling as use


def modify_module_state_dict_keys(msd):
    prefix = 'module.'
    for k in list(msd.keys()):
        if k.startswith(prefix):
            newk = k[len(prefix):]
            msd[newk] = msd[k]
            del msd[k]
    return msd


def create_ema_model(net, resume=False):
    # Network definition
    net_ema = copy.deepcopy(net)

    # load weights
    if resume:
        assert os.path.isfile(resume), f'EMA Model loading path is invalid!'
        checkpoint = torch.load(resume, map_location=torch.device('cpu'))
        msd = modify_module_state_dict_keys(checkpoint[f'model_state_dict'])
        net_ema.load_state_dict(msd)
        print(f"EMA model has been loaded from ckpt file!")
    else:
        print(f"EMA model has been initialized from student model!")

    # detach params
    for param in net_ema.parameters():
        param.detach_()
        
    return net_ema

# 用于更新指数移动平均（EMA）模型的参数。
def update_ema_variables(net, net_ema, alpha):
    # Use the true average until the exponential average is more correct
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(net_ema.parameters(), net.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
    return


# 软标签
def generate_soft_bounds(mask, filter_size):
    # make sure the shape of mask is [bs, ch, h, w]
    shp = mask.shape
    assert len(shp)==3 or len(shp)==4, 'Wrong size of mask in soft label generation!'
    if len(shp)==3: mask = torch.unsqueeze(mask, axis=1)  
    # filter    
    filters = torch.ones(1,1,filter_size,filter_size)
    new_masks = F.conv2d(mask, filters, padding='same')  # 这个卷积操作相当于对 mask 进行了平均池化。
    new_masks = new_masks/(filter_size**2)
    # reshape to original size
    if len(shp)==3: new_masks = torch.squeeze(new_masks,1)
    return new_masks

# 将轮廓绘制到一个 mask 上，每个轮廓的颜色都是不同的
def draw_contours_to_255(contours, mask_size):
    mask = np.zeros(list(mask_size)+[3], np.uint8)
    for ci, cnt in enumerate(contours):
        cv2.drawContours(mask, [cnt], 0, (ci+1,ci+1,ci+1), thickness=cv2.FILLED)
    mask = mask[:,:,0]
    return mask.astype(int)


def obj_wise_label_correction(ns_masks, pred_masks, filter_size=-1, filter_all=False):
    # 创建一个数据预处理的管道，这个管道将输入的数据转换为张量并进行归一化
    transform = transforms.Compose([transforms.ToTensor()])
    new_masks_add = []
    # 1 - update on each prediction map
    for pi in range(pred_masks.shape[0]):
        pred0 = pred_masks[pi].squeeze()
        ns0 = ns_masks[pi].squeeze().astype(float)
        
        # 1> - find contours
        # 找到图像中的轮廓
        contours, _ = cv2.findContours(pred0,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        n_objs = len(contours)

        # 2> - index buildings
        if n_objs<=255:
            mask = draw_contours_to_255(contours, pred0.shape)
        else:
            # 将一组轮廓绘制到一个 mask 上，并且每个轮廓都有一个唯一的标签
            n_mask = int(np.ceil(n_objs/255))
            mask = np.zeros(pred0.shape, np.int32)
            for mi in range(n_mask):
                obj_s = mi*255
                obj_e = min((mi+1)*255,n_objs)
                contour_tmp = [contours[k] for k in range(obj_s, obj_e)]
                mask_tmp = draw_contours_to_255(contour_tmp, pred0.shape)
                mask_tmp[mask_tmp>0] = (mask_tmp+obj_s)[mask_tmp>0]
                mask += mask_tmp
        
        # 3> - check overlap, remove objects overlapped with gt 移除 mask 中和 ns0 重叠的部分
        overlap_inds = np.unique(ns0*mask)
        for ind in overlap_inds: mask[mask==ind] = 0
        mask = (mask!=0).astype(np.float32)
        
        # 5> - merge gt with predictions
        new_masks_add.append(mask)
    
    # 2 - convert np.array to torch.tensor
    new_masks_add = transform(np.dstack(new_masks_add))
    old_masks_t = transform(ns_masks.astype(np.float32).transpose(1,2,0))
    
    # 3 - new noisy labels
    if filter_size>0:
        # 1> convert hard labels to soft ones
        if filter_all:
            # opt 1 - convert all the instances
            # 1 | update noisy labels
            new_masks = old_masks_t + new_masks_add
            y = torch.ones_like(new_masks)
            new_masks = torch.where(new_masks <= 1, new_masks, y)
            # 2 | generate soft labels
            new_masks = generate_soft_bounds(new_masks, filter_size)
        else:
            # opt 2 - only convert the corrected ones
            # 1 | generate soft labels for corrected ones
            new_masks_add = generate_soft_bounds(new_masks_add, filter_size)
            # 2 | update noisy labels
            new_masks = old_masks_t + new_masks_add
            y = torch.ones_like(new_masks)
            new_masks = torch.where(new_masks <= 1, new_masks, y)
    else:
        # 2> - use hard labels
        new_masks = old_masks_t + new_masks_add
        y = torch.ones_like(new_masks)
        new_masks = torch.where(new_masks <= 1, new_masks, y)
    
    return new_masks


def pixel_wise_label_correction(ns_masks, pred_logits, confidence=0.8):
    pred_probs = torch.sigmoid(pred_logits)
    ns_masks[pred_probs>confidence] = 1
    ns_masks[pred_probs<1-confidence] = 0
    return ns_masks


# # # # # # # # # # # #   consistency loss   # # # # # # # # # # # #
def mse_loss(input_logits, target_logits):
    """
    Takes softmax on both sides as inputs and returns MSE loss (mean)
    """
    assert input_logits.size() == target_logits.size()
    bs = input_logits.size()[0]
    if (input_logits.size())==3:
        input_softmax = torch.sigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
        n_class = 1
    else:
        input_softmax = F.softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
        n_class = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, reduction='mean')
    # return F.mse_loss(input_softmax, target_softmax, reduction='sum')/(bs*n_class)


def kl_loss(input_logits, target_logits, temperature=1):
    """
    Takes softmax on both sides as inputs and returns KL divergence
    """
    assert input_logits.size() == target_logits.size()
    if (input_logits.size())==3:
        input_log_softmax = F.logsigmoid(input_logits)
        target_softmax = torch.sigmoid(target_logits)
    else:
        input_log_softmax = F.log_softmax(input_logits, dim=1)
        target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, reduction='mean')*(temperature**2)


# # # # # # # # # # # #   ramp up   # # # # # # # # # # # #
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))



if __name__ == '__main__':
    plt.close('all')
    ind_dir1 = 'C:\\liuchy\\Research\\Projects\\Datasets\\Building_QY\\mass\\train\\ns_seg_rm_5'
    ind_dir2 = 'C:\\liuchy\\Research\\Projects\\Datasets\\Building_QY\\mass\\train\\ns_seg'
    n = 20
    fnames = os.listdir(ind_dir1)
    cands1 = []
    cands2 = []
    for fi in range(n):
        fname = fnames[fi]     
        cands1.append(cv2.imread(os.path.join(ind_dir1,fname),0))
        cands2.append(cv2.imread(os.path.join(ind_dir2,fname),0))
    
    cands1 = np.dstack(cands1).transpose(2,0,1)
    cands2 = np.dstack(cands2).transpose(2,0,1)
    
    # hard labels
    new_masks = obj_wise_label_correction(cands1, cands2, filter_size=5).numpy()
    for i in range(n):
        plt.figure(figsize=(12,4))
        plt.subplot(131)
        plt.imshow(cands1[i],interpolation='none')
        plt.subplot(132)
        plt.imshow(cands2[i],interpolation='none')
        plt.subplot(133)
        plt.imshow(new_masks[i],interpolation='none')
        
        
        
        
        
        