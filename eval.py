import os
from PIL import Image
import time
import logging
from argparse import ArgumentParser
import numpy as np
import yaml

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.nn.functional as F

from pyseg.models.model_helper import ModelBuilder
from pyseg.utils.utils import AverageMeter, intersectionAndUnion
from pyseg.utils.utils import check_makedirs, convert_state_dict, colorize


# Setup Parser
def get_parser():
    parser = ArgumentParser(description='PyTorch Evaluation')
    parser.add_argument(
        '--citywhole', action="store_true", default=False, help="whether crop evaluation for city"
    )
    parser.add_argument(
        '--base_size', type=int,
        default=2048, help='based size for scaling')
    parser.add_argument(
        '--scales', type=float,
        default=[0.5,0.75,1.0,1.25,1.5,2.0], nargs='+', help='evaluation scales')
    parser.add_argument(
        "--config", type=str, default="config.yaml")
    parser.add_argument(
        '--model_path', type=str,
        default='checkpoints/psp_best.pth', help='evaluation model path')
    parser.add_argument(
        '--save_folder', type=str,
        default='checkpoints/results/', help='results save folder')
    parser.add_argument(
        '--names_path', type=str,
        default='../../vis_meta/cityscapes/cityscapesnames.mat',
        help='path of dataset category names')
    parser.add_argument(
        '--crop', action="store_true", default=False, help="whether use crop evaluation"
    )
    return parser


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger

flag=0
count = 0
t = 0

def main():
    global args, logger, cfg
    args = get_parser().parse_args()
    cfg = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    logger = get_logger()
    logger.info(args)
    cfg_dset = cfg['dataset']
    mean, std = cfg_dset['mean'], cfg_dset['std']
    num_classes = cfg['net']['num_classes']
    crop_size = cfg_dset['val']['crop']['size']
    resize2 = cfg_dset.get('resize2', False)
    resize4 = cfg_dset.get('resize4', False)
    crop_h, crop_w = crop_size

    assert num_classes > 1

    gray_folder = os.path.join(args.save_folder, 'gray')
    color_folder = os.path.join(args.save_folder, 'color')

    cfg_dset = cfg['dataset']
    data_root, f_data_list = cfg_dset['val']['data_root'], cfg_dset['val']['data_list']
    data_list = []
    for line in open(f_data_list, 'r'):
        arr = line.strip().split(" ")
        arr = [os.path.join(data_root, item) for item in arr]
        if resize2:
            imp = arr[0].replace('leftImg8bit','leftImg8bit_2')
            labp = arr[1].replace('gtFine','gtFine_2')
            arr = [imp, labp]
        if resize4:
            imp = arr[0].replace('leftImg8bit','leftImg8bit_4')
            labp = arr[1].replace('gtFine','gtFine_4')
            arr = [imp, labp]
        data_list.append(arr)

    # Create network.
    args.use_auxloss = True if cfg['net'].get('aux_loss', False) else False
    logger.info("=> creating model ...")
    cfg['net']['sync_bn'] = False
    model = ModelBuilder(cfg['net'])
    saved_state_dict = convert_state_dict(torch.load(args.model_path)['model_state'])
    model.load_state_dict(saved_state_dict)
    model.cuda()
    logger.info("Load Model Done: {}".format(model))

    if cfg['dataset']['type'] == "cityscapes" and not args.citywhole:
        validate_city(model, num_classes, data_list, mean, std, args.base_size,
             crop_h, crop_w, args.scales, gray_folder, color_folder)
    else:
        valiadte_whole(model, num_classes, data_list, mean, std, args.scales, gray_folder, color_folder)
    cal_acc(data_list, gray_folder, num_classes)


def net_process(model, image):
    input = image.cuda()
    input_var = torch.autograd.Variable(input)

    output = model(input_var)[0] if args.use_auxloss else model(input_var)
    output = F.softmax(output, dim=1)
    return output


def scale_crop_process(model, image, classes, crop_h, crop_w, h, w, stride_rate=2/3):
    ori_h, ori_w = image.size()[-2:]
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        border = (pad_w_half, pad_w - pad_w_half, pad_h_half, pad_h - pad_h_half)
        image = F.pad(image, border, mode='constant', value=0.)
    new_h, new_w = image.size()[-2:]
    stride_h = int(np.ceil(crop_h*stride_rate))
    stride_w = int(np.ceil(crop_w*stride_rate))
    grid_h = int(np.ceil(float(new_h-crop_h)/stride_h) + 1)
    grid_w = int(np.ceil(float(new_w-crop_w)/stride_w) + 1)
    prediction_crop = torch.zeros((1, classes, new_h, new_w), dtype=torch.float).cuda()
    count_crop = torch.zeros((new_h, new_w), dtype=torch.float).cuda()
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[:, :, s_h:e_h, s_w:e_w].contiguous()
            count_crop[s_h:e_h, s_w:e_w] += 1
            '''
            with torch.no_grad():
                image_flip = flip_image(image_crop)
                flip_out = flip_image(net_process(model,image_flip))
                nor_out = net_process(model,image_crop)
                out = flip_out+nor_out
                prediction_crop[:, :, s_h:e_h, s_w:e_w] += out
            '''
            with torch.no_grad():  
                prediction_crop[:, :, s_h:e_h, s_w:e_w] += net_process(model, image_crop)
    prediction_crop /= count_crop
    prediction_crop = prediction_crop[:, :, pad_h_half:pad_h_half+ori_h, pad_w_half:pad_w_half+ori_w]
    prediction = F.interpolate(prediction_crop, size=(h, w), mode='bilinear', align_corners=True)
    return prediction[0]


def scale_whole_process(model, image, h, w):
    with torch.no_grad():
        prediction = net_process(model, image)
    prediction = F.interpolate(prediction, size=(h, w), mode='bilinear', align_corners=True)
    return prediction[0]


def validate_city(model, classes, data_list, mean, std, base_size, crop_h, crop_w, scales, gray_folder, color_folder):
    logger.info('>>>>>>>>>>>>>>>> Start Crop Evaluation >>>>>>>>>>>>>>>>')
    
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input_pth, _) in enumerate(data_list):
        data_time.update(time.time() - end)
        image = Image.open(input_pth).convert('RGB')
        image = np.asarray(image).astype(np.float32)
        image = (image - mean) / std
        image = torch.Tensor(image).permute(2, 0, 1)
        image = image.contiguous().unsqueeze(dim=0)
        h, w = image.size()[-2:]
        prediction = torch.zeros((classes, h, w), dtype=torch.float).cuda()
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size/float(h)*w)
            else:
                new_h = round(long_size/float(w)*h)
            image_scale = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=True)
            prediction += scale_crop_process(model, image_scale, classes, crop_h, crop_w, h, w)
        prediction = torch.max(prediction, dim=0)[1].cpu().numpy()
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(data_list),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = np.uint8(prediction)
        color = colorize(gray)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        gray = Image.fromarray(gray)
        gray.save(gray_path)
        color.save(color_path)
    logger.info('<<<<<<<<<<<<<<<<< End Crop Evaluation <<<<<<<<<<<<<<<<<')


def valiadte_whole(model, classes, data_list, mean, std, scales, gray_folder, color_folder):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input_pth, _) in enumerate(data_list):
        data_time.update(time.time() - end)
        image = Image.open(input_pth).convert('RGB')
        image = np.asarray(image).astype(np.float32)
        image = (image - mean) / std
        image = torch.Tensor(image).permute(2, 0, 1)
        image = image.contiguous().unsqueeze(dim=0)
        h, w = image.size()[-2:]
        prediction = torch.zeros((classes, h, w), dtype=torch.float).cuda()
        for scale in scales:
            new_h = round(h * scale)
            new_w = round(w * scale)
            image_scale = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=True)
            prediction += scale_whole_process(model, image_scale, h, w)
        prediction = torch.max(prediction, dim=0)[1].cpu().numpy()     ##############attention###############
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % 10 == 0:
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.format(i + 1, len(data_list),
                                                                                    data_time=data_time,
                                                                                    batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = np.uint8(prediction)
        color = colorize(gray)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        gray = Image.fromarray(gray)
        gray.save(gray_path)
        color.save(color_path)
    logger.info('<<<<<<<<<<<<<<<<< End  Evaluation <<<<<<<<<<<<<<<<<')


def cal_acc(data_list, pred_folder, classes):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    for i, (image_path, target_path) in enumerate(data_list):
        image_name = image_path.split('/')[-1].split('.')[0]
        pred = np.asarray(Image.open(os.path.join(pred_folder, image_name+'.png')))
        target = np.asarray(Image.open(target_path))
        if target.shape[-1] == 3:
            target = target[:,:,0]
        intersection, union, target = intersectionAndUnion(pred, target, classes)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)
        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        print('Evaluating {0}/{1} on image {2}, accuracy {3:.4f}.'
              .format(i + 1, len(data_list), image_name+'.png', accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    torch.save(mIoU, 'eval_metric.pth.tar')
    print('Eval result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    for i in range(classes):
        print('Class_{} result: iou/accuracy {:.4f}/{:.4f}'.format(i, iou_class[i], accuracy_class[i]))

def flip_image(img):
    assert(img.dim()==4)
    with torch.cuda.device_of(img):
        idx = torch.arange(img.size(3)-1, -1, -1).type_as(img).long()
    return img.index_select(3, idx)
    
if __name__ == '__main__':
    main()
