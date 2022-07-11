import os
import os.path
import numpy as np
import copy

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .base import BaseDataset
from . import augmentation as psp_trsform
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class city_dset(BaseDataset):
    def __init__(self, data_root, data_list, trs_form, resize=False, resize4=False):
        super(city_dset, self).__init__(data_list)
        self.data_root = data_root
        self.transform = trs_form
        self.resize = resize
        self.resize4 = resize4

    def __getitem__(self, index):
        # load image and its label
        image_path = os.path.join(self.data_root, self.list_sample[index][0])
        label_path = os.path.join(self.data_root, self.list_sample[index][1])
        if self.resize:
            image_path = image_path.replace('leftImg8bit','leftImg8bit_2')
            label_path = label_path.replace('gtFine','gtFine_2')
        if self.resize4:
            image_path = image_path.replace('leftImg8bit','leftImg8bit_4')
            label_path = label_path.replace('gtFine','gtFine_4')
        image = self.img_loader(image_path, 'RGB')
        label = self.img_loader(label_path, 'L')
        #print('image',image_path, image.shape)
        image, label = self.transform(image, label)
        return image[0], label[0, 0].long()


def build_transfrom(cfg):
    trs_form = []
    mean, std, ignore_label = cfg['mean'], cfg['std'], cfg['ignore_label']
    trs_form.append(psp_trsform.ToTensor())
    trs_form.append(psp_trsform.Normalize(mean=mean, std=std))
    if cfg.get('resize', False):
        trs_form.append(psp_trsform.Resize(cfg['resize']))
    if cfg.get('rand_resize', False):
        trs_form.append(psp_trsform.RandResize(cfg['rand_resize']))
    if cfg.get('rand_rotation', False):
        rand_rotation = cfg['rand_rotation']
        trs_form.append(psp_trsform.RandRotate(rand_rotation, ignore_label=ignore_label))
    if cfg.get('GaussianBlur', False) and cfg['GaussianBlur']:
        trs_form.append(psp_trsform.RandomGaussianBlur())
    if cfg.get('flip', False) and cfg.get('flip'):
        trs_form.append(psp_trsform.RandomHorizontalFlip())
    if cfg.get('crop', False):
        crop_size, crop_type = cfg['crop']['size'], cfg['crop']['type']
        trs_form.append(psp_trsform.Crop(crop_size, crop_type=crop_type, ignore_label=ignore_label))
    return psp_trsform.Compose(trs_form)


def build_cityloader(split, all_cfg):
    cfg_dset = all_cfg['dataset']
    cfg_trainer = all_cfg['trainer']

    cfg = copy.deepcopy(cfg_dset)
    cfg.update(cfg.get(split, {}))

    workers = cfg.get('workers', 2)
    batch_size = cfg.get('batch_size', 1)
    resize = cfg.get('resize2', False)
    resize4 = cfg.get('resize4', False)
    # build transform
    trs_form = build_transfrom(cfg)
    dset = city_dset(cfg['data_root'], cfg['data_list'], trs_form, resize=resize, resize4=resize4)

    # build sampler
    sample = DistributedSampler(dset)

    loader = DataLoader(dset, batch_size=batch_size, num_workers=workers,
                        sampler=sample, shuffle=False, pin_memory=False)
    
    return loader