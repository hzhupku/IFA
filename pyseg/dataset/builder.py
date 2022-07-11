
import logging

from .cityscapes import build_cityloader
from .pascal_voc import build_vocloader


logger = logging.getLogger('global')

def get_loader(cfg):
    cfg_dataset = cfg['dataset']
    if cfg_dataset['type'] == 'cityscapes':
        trainloader = build_cityloader('train', cfg)
        valloader = build_cityloader('val', cfg)
    elif cfg_dataset['type'] == 'pascal_voc':
        trainloader = build_vocloader('train', cfg)
        valloader = build_vocloader('val', cfg)
    else:
        raise NotImplementedError("dataset type {} is not supported".format(cfg_dataset))
    logger.info('Get loader Done...')
 
    return trainloader, valloader
