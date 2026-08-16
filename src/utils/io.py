'''
Function:
    Implementation of IO related operations
Author:
    Zhenchao Jin
'''
import os
import torch
import torch.utils.model_zoo as model_zoo
import json


'''judgefileexist'''
def judgefileexist(filepath):
    if os.path.islink(filepath):
        filepath = os.readlink(filepath)
    return os.path.exists(filepath)


'''touchdir'''
def touchdir(directory):
    if not os.path.exists(directory):
        try:
            os.mkdir(directory)
        except:
            pass


'''touchdirs'''
def touchdirs(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except:
            pass


'''loadckpts'''
def loadckpts(ckptspath, map_to_cpu=True):
    if os.path.islink(ckptspath):
        ckptspath = os.readlink(ckptspath)
    if map_to_cpu: 
        ckpts = torch.load(ckptspath, map_location=torch.device('cpu'))
    else: 
        ckpts = torch.load(ckptspath)
    return ckpts


'''saveckpts'''
def saveckpts(ckpts, savepath, make_soft_link=True, soft_link_dst=None):
    save_response = torch.save(ckpts, savepath)
    if make_soft_link:
        if soft_link_dst is None:
            soft_link_dst = os.path.join(os.path.dirname(savepath), 'checkpoints-epoch-latest.pth')
        symlink(savepath, soft_link_dst)
    return save_response


'''symlink'''
def symlink(src_path, dst_path):
    if os.path.islink(dst_path):
        os.unlink(dst_path)
    os.symlink(src_path, dst_path)
    return True


'''loadpretrainedweights'''
def loadpretrainedweights(structure_type, pretrained_model_path='', default_model_urls={}, map_to_cpu=True, possible_model_keys=['model', 'state_dict']):
    checkpoint = None
    cache_dir = pretrained_model_path if os.path.isdir(pretrained_model_path) else ''
    registry_path = os.path.join(cache_dir, 'registries.json') if cache_dir else ''

    if pretrained_model_path and os.path.isfile(pretrained_model_path):
        checkpoint = torch.load(
            pretrained_model_path,
            map_location='cpu' if map_to_cpu else None,
        )
    elif registry_path and os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registries = json.load(f)
        if structure_type in registries:
            cached_model_path = os.path.join(cache_dir, registries[structure_type])
            if os.path.exists(cached_model_path):
                print(f'找到缓存模型: {cached_model_path}')
                checkpoint = torch.load(
                    cached_model_path,
                    map_location='cpu' if map_to_cpu else None,
                )

    if checkpoint is None:
        if structure_type in default_model_urls:
            print(f'缓存目录不存在或未找到模型，尝试在线下载: {pretrained_model_path}')
            checkpoint = model_zoo.load_url(default_model_urls[structure_type],
                model_dir=cache_dir or None,
                map_location='cpu' if map_to_cpu else None,
                progress=True
            )
            if cache_dir:
                try:
                    registries = {}
                    if os.path.exists(registry_path):
                        with open(registry_path, 'r') as f:
                            registries = json.load(f)
                    registries[structure_type] = os.path.basename(default_model_urls[structure_type])
                    with open(registry_path, 'w') as f:
                        json.dump(registries, f, indent=4)
                except (OSError, ValueError) as exc:
                    print(f'更新注册表失败: {exc}')
        else:
            print(f'无法找到预训练权重: 结构类型 {structure_type} 不在默认模型 URL 中')
            return None

    # 提取 state_dict
    state_dict = checkpoint
    for key in possible_model_keys:
        if key in checkpoint:
            state_dict = checkpoint[key]
            break
    return state_dict
