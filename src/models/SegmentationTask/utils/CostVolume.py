import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.SegmentationTask.utils.block import *
from src.models.SegmentationTask.utils.attentions import CAM, Change_CAM

class CostVolumeLayer(nn.Module):

    def __init__(self, search_range=2):
        super(CostVolumeLayer, self).__init__()
        self.search_range = search_range

    def forward(self, x1, x2):

        shape = list(x1.size()); shape[1] = (self.search_range * 2 + 1) ** 2
        cv = torch.zeros(shape).cuda()

        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                if   i < 0: slice_h, slice_h_r = slice(None, i), slice(-i, None)
                elif i > 0: slice_h, slice_h_r = slice(i, None), slice(None, -i)
                else:       slice_h, slice_h_r = slice(None),    slice(None)

                if   j < 0: slice_w, slice_w_r = slice(None, j), slice(-j, None)
                elif j > 0: slice_w, slice_w_r = slice(j, None), slice(None, -j)
                else:       slice_w, slice_w_r = slice(None),    slice(None)

                cv[:, (self.search_range*2+1) * i + j, slice_h, slice_w] = (x1[:,:,slice_h, slice_w]  * x2[:,:,slice_h_r, slice_w_r]).sum(1)
    
        return cv / shape[1]


class CostVolumeLayer2(nn.Module):

    def __init__(self, search_range=2):
        super(CostVolumeLayer2, self).__init__()
        self.search_range = search_range

    def forward(self, x1, x2):
        x1 = torch.sigmoid(x1)
        x2 = torch.sigmoid(x2)
        shape = list(x1.size()); shape[1] = (self.search_range * 2 + 1) ** 2
        cv = x1[:,0:1,::].expand(shape)*0

        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                # if   i < 0: slice_h, slice_h_r = slice(None, i), slice(-i, None)
                # elif i > 0: slice_h, slice_h_r = slice(i, None), slice(None, -i)
                # else:       slice_h, slice_h_r = slice(None),    slice(None)

                # if   j < 0: slice_w, slice_w_r = slice(None, j), slice(-j, None)
                # elif j > 0: slice_w, slice_w_r = slice(j, None), slice(None, -j)
                # else:       slice_w, slice_w_r = slice(None),    slice(None)
                if i < 0:
                    if j < 0:
                        cv[:, (self.search_range*2+1) * i + j, 0:shape[2]+i, 0:shape[3]+j] = torch.mul(x1[:,:,0:shape[2]+i, 0:shape[3]+j], x2[:,:,-i:, -j:]).sum(1)
                    elif j == 0:
                        cv[:, (self.search_range*2+1) * i + j, 0:shape[2]+i, :] = torch.mul(x1[:,:,0:shape[2]+i, :], x2[:,:,-i:, :]).sum(1)
                    else:
                        cv[:, (self.search_range*2+1) * i + j, 0:shape[2]+i, j:] = torch.mul(x1[:,:,0:shape[2]+i, j:], x2[:,:,-i:, 0:shape[3]-j]).sum(1)
                elif i == 0:
                    if j < 0:
                        cv[:, j, :, 0:shape[3]+j] = torch.mul(x1[:,:,:, 0:shape[3]+j], x2[:,:,:, -j:]).sum(1)
                    elif j == 0:
                        cv[:, j, :, :] = torch.mul(x1[:,:,:, :], x2[:,:,:, :]).sum(1)
                    else:
                        cv[:, j, :, j:] = torch.mul(x1[:,:,:, j:], x2[:,:,:, 0:shape[3]-j]).sum(1)
                else:
                    if j < 0:
                        cv[:, (self.search_range*2+1) * i + j, i:, 0:shape[3]+j] = torch.mul(x1[:,:,i:, 0:shape[3]+j], x2[:,:,0:shape[2]-i, -j:]).sum(1)
                    elif j == 0:
                        cv[:, (self.search_range*2+1) * i + j, i:, :] = torch.mul(x1[:,:,i:, :], x2[:,:,0:shape[2]-i, :]).sum(1)
                    else:
                        cv[:, (self.search_range*2+1) * i + j, i:, j:] = torch.mul(x1[:,:,i:, j:], x2[:,:,0:shape[2]-i, 0:shape[3]-j]).sum(1)

        return cv / shape[1]

class CostVolumeLayer3(nn.Module):

    def __init__(self, search_range=2):
        super(CostVolumeLayer3, self).__init__()
        self.search_range = search_range
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        shape = list(x1.shape)
        cv = x1[:,0:1,::]*0
        x1 =  F.pad(self.sigmoid(x1), (self.search_range, self.search_range, self.search_range, self.search_range), mode="constant")
        x2 =  F.pad(self.sigmoid(x2), (self.search_range, self.search_range, self.search_range, self.search_range), mode="constant")
        shape[1] = (self.search_range * 2 + 1) ** 2
        cv = cv.expand(shape)

        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                shift1 = [i, j]
                shift2 = [-i, -j]
                x1_new = torch.roll(x1, shifts=shift1, dims=[2,3])
                x2_new = torch.roll(x2, shifts=shift2, dims=[2,3])
                cv[:, (self.search_range*2+1) * i + j, ::] = torch.mul(x1_new, x2_new).sum(1)[:,self.search_range:shape[2]+self.search_range,self.search_range:shape[3]+self.search_range]
        return cv / shape[1]

class CDM(nn.Module):
    def __init__(self, channel):
        super(CDM, self).__init__()
        self.conv_1x1_1 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )
        self.conv_1x1_2 = nn.Sequential(
            nn.Conv2d(channel*2, channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel),
            nn.ReLU(),
        )

    def forward(self, f1, f2):
        fd1 = f1 - f2
        f12 = torch.cat((f1, f2), 1)
        fd2 = self.conv_1x1_1(f12)
        fd = self.conv_1x1_2(torch.cat((fd1, fd2), 1))
        return fd
        

class CCAM(nn.Module):
    """ Change Channel attention module"""
    def __init__(self, in_dim):
        super(CCAM, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_dim, in_dim*3, kernel_size=1, stride=1, padding=0)
        self.cdm = CDM(in_dim)
        self.softmax  = nn.Softmax(dim=-1)
        # self.bn = nn.BatchNorm2d(in_dim)
        # self.gamma1 = nn.Parameter(torch.zeros(1))
        # self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x1.size()
        diff = self.cdm(x1, x2)
        qkv = self.conv_1x1(diff).reshape(m_batchsize, C, 3, height, width).permute(2, 0, 1, 3, 4)
        proj_query, proj_key, proj_value = qkv[0], qkv[1], qkv[2]
        proj_query = proj_query.view(m_batchsize, C, -1)
        proj_key = proj_key.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_value = proj_value.view(m_batchsize, C, -1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy

        attention =self.softmax(energy_new)

        proj_value1 = x1.view(m_batchsize, C, -1)
        proj_value2 = x2.view(m_batchsize, C, -1)

        out1 = torch.bmm(attention, proj_value1)
        out1 = out1.view(m_batchsize, C, height, width)
        out2 = torch.bmm(attention, proj_value2)
        out2 = out2.view(m_batchsize, C, height, width)
        out_diff = torch.bmm(attention, proj_value)
        out_diff = out_diff.view(m_batchsize, C, height, width)
        out1 = out1 + x1
        out2 = out2 + x2
        out_diff = out_diff + diff
        
        return out1, out2, out_diff


class CCAM2(nn.Module):
    """ Change Channel attention module"""
    def __init__(self, in_dim, search_range):
        super(CCAM2, self).__init__()
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_dim),
        )
        self.conv_qkv = nn.Conv2d(in_dim, in_dim*3, kernel_size=1, stride=1, padding=0)
        self.relu  = nn.ReLU()
        self.softmax  = nn.Softmax(dim=-1)
        self.sim_module = RegionSimilarity(in_dim, search_range)

    def forward(self, x1, x2):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x1.size()
        diff = self.relu(self.conv_1x1(torch.pow(x1 - x2, 2)))
        qkv = self.conv_qkv(diff).reshape(m_batchsize, C, 3, height, width).permute(2, 0, 1, 3, 4)
        proj_query, proj_key, proj_value = qkv[0], qkv[1], qkv[2]
        proj_query = proj_query.view(m_batchsize, C, -1)
        proj_key = proj_key.view(m_batchsize, C, -1).permute(0, 2, 1)
        proj_value = proj_value.view(m_batchsize, C, -1)

        energy = torch.bmm(proj_query, proj_key)
        """way 1"""
        # energy_new = (torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy)
        # """way 2"""
        energy_new = energy / C ** 0.5
        # """way 3"""
        # energy_new = torch.sqrt(energy/C) #cause nan

        attention =self.softmax(energy_new)

        proj_value1 = x1.view(m_batchsize, C, -1)
        proj_value2 = x2.view(m_batchsize, C, -1)

        out1 = torch.bmm(attention, proj_value1)
        out1 = out1.view(m_batchsize, C, height, width)
        out2 = torch.bmm(attention, proj_value2)
        out2 = out2.view(m_batchsize, C, height, width)
        out_diff = torch.bmm(attention, proj_value)
        out_diff = out_diff.view(m_batchsize, C, height, width)
        out1 = self.relu(out1 + x1)
        out2 = self.relu(out2 + x2)
        out_diff = self.relu(out_diff + diff)
        
        return out1, out2, out_diff


class CV9(nn.Module):
    def __init__(self, dim_in, dim_out, search_range=1):
        super(CV9, self).__init__()
        self.ccam = CCAM2(dim_in, search_range)
        self.search_range = search_range
        self.cv_layer = CostVolumeLayer2(self.search_range)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d((self.search_range * 2 + 1) ** 2, self.search_range * 2 + 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.search_range * 2 + 1),
            nn.Conv2d(self.search_range * 2 + 1, 1, kernel_size=1, stride=1, padding=0),
        )
        self.cat = cat_conv(dim_in, dim_in, dim_in, False)
        self.conv_1x1_2 = nn.Sequential(
            nn.Conv2d(dim_in*2, dim_in, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
        )
        
    def forward(self, x1, x2):
        x1, x2, diff = self.ccam(x1, x2)
        cv = 1 - self.cv_layer(x1, x2) / (self.search_range * 2 + 1) ** 2
        cv = self.conv_1x1(cv)
        cv = torch.sigmoid(cv)
        y = self.cat(x1, x2)
        y = self.conv_1x1_2(torch.cat((y, diff), 1))
        # from clcore import ImageIO;io = ImageIO()
        # aa = y[0].permute(1,2,0)
        # io.write_image(f'./test1.tif', aa.cpu().numpy(), dtype='float32')
        # import ipdb;ipdb.set_trace()
        # import cv2;import numpy as np
        # aa = cv[0][0].cpu().numpy()*255
        # # cv2.imwrite('test.png', aa.astype(np.uint8))
        # aa = cv2.resize(aa, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        # aa = aa.astype(np.uint8)
        # heatmap = cv2.applyColorMap(aa, cv2.COLORMAP_JET)
        # cv2.imwrite('test110_hm.png', heatmap)
        # import ipdb;ipdb.set_trace()
        return y, cv


class CV10(CV9):
    def __init__(self, dim_in, dim_out, search_range=1):
        super(CV10, self).__init__(dim_in, dim_out, search_range)
        self.cv_layer = CostVolumeLayer3(self.search_range)
        
    def forward(self, x1, x2):
        x1, x2, diff = self.ccam(x1, x2)
        cv = 1 - self.cv_layer(x1, x2) / (self.search_range * 2 + 1) ** 2
        cv = self.conv_1x1(cv)
        cv = torch.sigmoid(cv)
        y = self.cat(x1, x2)
        y = self.conv_1x1_2(torch.cat((y, diff), 1))
        return y, cv


"""Region similarty module with cosine similarity"""
"""
For: bitemporal features have similar marginal distribution but vary in the mean value.
In this case, the euclidean metric is big but the matrix of cosine similarity is small. 
"""
class RegionSimilarity(nn.Module):
    def __init__(self, channel, search_range=2):
        super(RegionSimilarity, self).__init__()
        self.search_range = search_range
        # self.conv_1x1 = nn.Sequential(
        #     nn.Conv2d(channel, channel//2, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(channel//2),
        #     nn.ReLU(),
        #     nn.Conv2d(channel//2, 1, kernel_size=3, stride=1, padding=1),
        # )

    def forward(self, x1, x2):
        shape = list(x1.shape)
        sim = torch.zeros((shape[0], (self.search_range * 2 + 1) ** 2, shape[2], shape[3])).to(x1.device)
        tmp_x = torch.cat((x1, x2), 1)
        tmp_x = torch.softmax(tmp_x, 1)*2
        x1 = tmp_x[:,0:shape[1],::]
        x2 = tmp_x[:,shape[1]:,::]

        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                shift1 = [i, j]
                shift2 = [-i, -j]
                x1_new = torch.roll(x1, shifts=shift1, dims=[2,3])
                x2_new = torch.roll(x2, shifts=shift2, dims=[2,3])
                # cos_d = F.cosine_similarity(x1_new, x2_new, dim=1)[:,self.search_range:shape[2]-self.search_range,self.search_range:shape[3]-self.search_range]
                # eulid = 1 - F.pairwise_distance(x1_new, x2_new, p=2)[:,self.search_range:shape[2]+self.search_range,self.search_range:shape[3]+self.search_range] / x1.size(1)
                rs = torch.mul(x1_new, x2_new).sum(1)[:,abs(i):shape[2]-abs(i),abs(j):shape[3]-abs(j)]
                # rs = F.pad(rs.unsqueeze(1), (abs(j), abs(j), abs(i), abs(i)), mode="replicate")
                # rs = self.conv_1x1(x1_new - x2_new)[:,0,abs(i):shape[2]-abs(i),abs(j):shape[3]-abs(j)]
                sim[:, (self.search_range*2+1) * i + j, abs(i):shape[2]-abs(i),abs(j):shape[3]-abs(j)] = rs
        return sim


class RegionSimilarity2(RegionSimilarity):
    def __init__(self, search_range=2):
        super(RegionSimilarity2, self).__init__(search_range)

    def forward(self, x1, x2):
        shape = list(x1.shape)
        shape[1] = (self.search_range * 2 + 1) ** 2
        sim = torch.zeros(shape).to(x1.device)
        x1 = torch.softmax(x1, 1)
        x2 = torch.softmax(x2, 1)

        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                if i < 0:
                    if j < 0:
                        sim[:, (self.search_range*2+1) * i + j, 0:shape[2]+i, 0:shape[3]+j] = torch.mul(x1[:,:,0:shape[2]+i, 0:shape[3]+j], x2[:,:,-i:, -j:]).sum(1)
                    elif j == 0:
                        sim[:, (self.search_range*2+1) * i + j, 0:shape[2]+i, :] = torch.mul(x1[:,:,0:shape[2]+i, :], x2[:,:,-i:, :]).sum(1)
                    else:
                        sim[:, (self.search_range*2+1) * i + j, 0:shape[2]+i, j:] = torch.mul(x1[:,:,0:shape[2]+i, j:], x2[:,:,-i:, 0:shape[3]-j]).sum(1)
                elif i == 0:
                    if j < 0:
                        sim[:, j, :, 0:shape[3]+j] = torch.mul(x1[:,:,:, 0:shape[3]+j], x2[:,:,:, -j:]).sum(1)
                    elif j == 0:
                        sim[:, j, :, :] = torch.mul(x1[:,:,:, :], x2[:,:,:, :]).sum(1)
                    else:
                        sim[:, j, :, j:] = torch.mul(x1[:,:,:, j:], x2[:,:,:, 0:shape[3]-j]).sum(1)
                else:
                    if j < 0:
                        sim[:, (self.search_range*2+1) * i + j, i:, 0:shape[3]+j] = torch.mul(x1[:,:,i:, 0:shape[3]+j], x2[:,:,0:shape[2]-i, -j:]).sum(1)
                    elif j == 0:
                        sim[:, (self.search_range*2+1) * i + j, i:, :] = torch.mul(x1[:,:,i:, :], x2[:,:,0:shape[2]-i, :]).sum(1)
                    else:
                        sim[:, (self.search_range*2+1) * i + j, i:, j:] = torch.mul(x1[:,:,i:, j:], x2[:,:,0:shape[2]-i, 0:shape[3]-j]).sum(1)
        return sim

"""Region Fusion Module"""
class RFM(CV10):
    def __init__(self, dim_in, dim_out, search_range=1, deep_supervise=False):
        super(RFM, self).__init__(dim_in, dim_out, search_range)
        self.deep_supervise = deep_supervise
        self.sim_module = RegionSimilarity(dim_in, self.search_range)
        self.ccam = CCAM2(dim_in, self.search_range)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d((self.search_range * 2 + 1) ** 2, self.search_range * 2 + 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.search_range * 2 + 1),
            nn.ReLU(),
            nn.Conv2d(self.search_range * 2 + 1, dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(dim_out),
        )
        if self.deep_supervise:
            self.conv_aux = nn.Sequential(
                nn.Conv2d(dim_out, 1, kernel_size=3, stride=1, padding=1),
                nn.Sigmoid(),
            )
        
    def forward(self, x1, x2):
        x1, x2, diff = self.ccam(x1, x2)
        dist = 1 - self.sim_module(x1, x2)
            # aux_out = torch.min(dist, dim=1, keepdim=True)[0]
        # dist = torch.min(dist, dim=1, keepdim=True)[0]
        dist = self.conv_1x1(dist)
        if self.deep_supervise:
            aux_out = self.conv_aux(dist)
        y = self.cat(x1, x2)
        y = self.conv_1x1_2(torch.cat((y, diff), 1))
        # from clcore import ImageIO;io = ImageIO()
        # aa = x1[0].permute(1,2,0)
        # io.write_image(f'/nfs/project/netdisk/192.168.10.227/d/cp/ST/out/RFNet_DS_dice_explr18/sft0/test1.tif', aa.cpu().numpy(), dtype='float32')
        # aa = x2[0].permute(1,2,0)
        # io.write_image(f'/nfs/project/netdisk/192.168.10.227/d/cp/ST/out/RFNet_DS_dice_explr18/sft0/test2.tif', aa.cpu().numpy(), dtype='float32')
        # aa = dist[0].permute(1,2,0)
        # io.write_image(f'/nfs/project/netdisk/192.168.10.227/d/cp/ST/out/RFNet_DS_dice_explr18/sft0/dist.tif', aa.cpu().numpy(), dtype='float32')
        # aa = (1 - self.sim_module(x1, x2))[0].permute(1,2,0)
        # io.write_image(f'/nfs/project/netdisk/192.168.10.227/d/cp/ST/out/RFNet_DS_dice_explr18/sft0/rs.tif', aa.cpu().numpy(), dtype='float32')
        # import cv2;import numpy as np
        # aa = aa.min(2)[0].cpu().numpy()
        # aa = cv2.normalize(aa, None, 0, 255, cv2.NORM_MINMAX)
        # aa = cv2.resize(aa, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        # aa = aa.astype(np.uint8)
        # heatmap = cv2.applyColorMap(aa, cv2.COLORMAP_JET)
        # cv2.imwrite('/nfs/project/netdisk/192.168.10.227/d/cp/ST/out/RFNet_DS_dice_explr18/sft0/rs_min.png', heatmap)
        # # aa = torch.abs(x1-x2)[0].sum(0).cpu().numpy()
        # aa = cv2.normalize(aa, None, 0, 255, cv2.NORM_MINMAX)
        # aa = cv2.resize(aa, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        # aa = aa.astype(np.uint8)
        # heatmap = cv2.applyColorMap(aa, cv2.COLORMAP_JET)
        # cv2.imwrite('/nfs/project/netdisk/192.168.10.227/d/cp/ST/out/RFNet_DS_dice_explr14/sft0/test110_hm_diff.png', heatmap)
        # aa = dist[0].min(0)[0].cpu().numpy()
        # aa = cv2.normalize(aa, None, 0, 255, cv2.NORM_MINMAX)
        # aa = cv2.resize(aa, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        # aa = aa.astype(np.uint8)
        # heatmap = cv2.applyColorMap(aa, cv2.COLORMAP_JET)
        # cv2.imwrite('/nfs/project/netdisk/192.168.10.227/d/cp/ST/out/RFNet_DS_dice_explr14/sft0/test110_hm_dist.png', heatmap)
        # import ipdb;ipdb.set_trace()
        # import sys;sys.exit(0)
        if self.deep_supervise:
            return y, dist, aux_out
        else:
            return y, dist


class CCAM3(nn.Module):
    def __init__(self, channels):
        super(CCAM3, self).__init__()
        self.cam = CAM(channels)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
    
    def forward(self, x1, x2):
        diff = self.conv_1x1(x1 - x2)
        cw = self.cam(diff)
        x1 = x1 * cw + x1
        x2 = x1 * cw + x2
        return x1, x2, diff


class RegionSimilarity3(nn.Module):
    def __init__(self, channel, search_range=2, stride=1):
        super(RegionSimilarity3, self).__init__()
        self.search_range = search_range
        self.conv_kv = nn.Conv2d(channel * (self.search_range * 2 + 1) ** 2, channel*2, kernel_size=1, stride=1, padding=0)
        self.conv_q = nn.Conv2d(channel * (self.search_range * 2 + 1) ** 2, channel, kernel_size=1, stride=1, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.stride = stride
        self.pool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        self.conv_proj = nn.Sequential(
            nn.Conv2d(channel//(stride**2), channel//(stride**2), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channel//(stride**2)),
            nn.ReLU(),
            nn.Conv2d(channel//(stride**2), 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        m_batchsize, C, height, width = x1.size()
        x1_list = []
        x2_list = []
        for i in range(-self.search_range, self.search_range + 1):
            for j in range(-self.search_range, self.search_range + 1):
                shift1 = [i, j]
                shift2 = [-i, -j]
                sft1 = torch.roll(x1, shifts=shift1, dims=[2,3])[:,:,abs(i):height-abs(i),abs(j):width-abs(j)]
                sft1 = F.pad(sft1, (abs(j), abs(j), abs(i), abs(i)), mode="replicate")
                x1_list.append(sft1)
                sft2 = torch.roll(x2, shifts=shift2, dims=[2,3])[:,:,abs(i):height-abs(i),abs(j):width-abs(j)]
                sft2 = F.pad(sft2, (abs(j), abs(j), abs(i), abs(i)), mode="replicate")
                x2_list.append(sft2)
        x1_new = torch.cat(x1_list, 1)
        x2_new = torch.cat(x2_list, 1)
        if self.stride > 1:
            x1_new = self.pool(x1_new)
            x2_new = self.pool(x2_new)
        kv = self.conv_kv(x1_new).reshape(m_batchsize, C, 2, height//self.stride, width//self.stride).permute(2, 0, 1, 3, 4)
        key, value = kv[0], kv[1]
        key = key.view(m_batchsize, C, -1)
        value = value.view(m_batchsize, C, -1).permute(0, 2, 1)

        query = self.conv_q(x2_new).view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(query, key)#B*HW*HW
        """way 1"""
        # energy_new = (torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy)
        # """way 2"""
        energy_new = energy / C ** 0.5
        # """way 3"""
        # energy_new = torch.sqrt(energy/C) #cause nan

        attention = self.softmax(energy_new)

        out = torch.bmm(attention, value)#B*HW*C
        out = out.view(m_batchsize, C//(self.stride**2), height, width)
        
        return self.conv_proj(out)



class RFM2(nn.Module):
    def __init__(self, dim_in, dim_out, search_range=1, deep_supervise=False, stride=1):
        super(RFM2, self).__init__()
        
        self.dim_out = dim_out
        self.deep_supervise = deep_supervise
        self.ccam = CCAM3(dim_in)
        self.sim_module = RegionSimilarity3(dim_in, search_range, stride)
        self.cat = cat_conv(dim_in, dim_in, dim_in, False)
        self.conv_1x1 = nn.Sequential(
            nn.Conv2d(dim_in*2, dim_in, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim_in),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        bs, C, height, width = x1.size()
        x1, x2, diff = self.ccam(x1, x2)
        dist = self.sim_module(x1, x2)
        y = self.cat(x1, x2)
        y = self.conv_1x1(torch.cat((y, diff), 1))
        if self.deep_supervise:
            return y, dist, dist
        else:
            return y, dist
__all__ = [
    "CostVolumeLayer",
    "RFM",
    "RFM2",
]