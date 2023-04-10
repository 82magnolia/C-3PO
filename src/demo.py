import datetime
import os
import time
import math

import torch
import torch.utils.data

from collections import OrderedDict
import numpy as np

import utils

from dataset import dataset_dict
from model import model_dict
import loss
import dataset.transforms as T 
from PIL import Image
from dataset.transforms import proc_image
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import cv2


def get_scheduler_function(name, total_iters, final_lr=0):
    print("LR Scheduler: {}".format(name))
    if name == 'cosine':
        return lambda step: ((1 + math.cos(step * math.pi / total_iters)) / 2) * (1 - final_lr) + final_lr
    elif name == 'linear':
        return lambda step: 1 - (1 - final_lr) / total_iters * step
    elif name == 'exp':
        return lambda step: (1 - step / total_iters) ** 0.9
    elif name == 'none':
        return lambda step: 1
    else:
        raise ValueError(name)


def warmup(num_iter, num_warmup, optimizer):
    if num_iter < num_warmup:
        # warm up
        xi = [0, num_warmup]
        for j, x in enumerate(optimizer.param_groups):
            # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
            #x['lr'] = np.interp(num_iter, xi, [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
            x['lr'] = np.interp(num_iter, xi, [0.1 if j == 2 else 0.0, x['lr']])
            if 'momentum' in x:
                x['momentum'] = np.interp(num_iter, xi, [0.8, 0.9])


def fix_BN_stat(module):
    classname = module.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        module.eval()
    #if classname.find('LayerNorm') != -1:
    #    module.eval()


def freeze_BN_stat(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        if hasattr(model.module, 'backbone'):
            print("freeze backbone BN stat")
            model.module.backbone.apply(fix_BN_stat)


def create_dataloader(args):
    dataset = dataset_dict[args.train_dataset](args, train=True)
    dataset_test = dataset_dict[args.test_dataset](args, train=False)
    if args.test_dataset2:
        dataset_test2 = dataset_dict[args.test_dataset2](args, train=False)
    else:
        dataset_test2 = None
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
        test_sampler2 = torch.utils.data.distributed.DistributedSampler(dataset_test2) if dataset_test2 else None
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)
        test_sampler2 = torch.utils.data.SequentialSampler(dataset_test2) if dataset_test2 else None

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    if dataset_test2:
        data_loader_test2 = torch.utils.data.DataLoader(
            dataset_test2, batch_size=1,
            sampler=test_sampler2, num_workers=args.workers,
            collate_fn=utils.collate_fn)
    else:
        data_loader_test2 = None

    return dataset, train_sampler, data_loader, dataset_test, data_loader_test, dataset_test2, data_loader_test2
    

def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    args.num_classes = 2
    print(args)
    
    device = torch.device(args.device)

    model = model_dict[args.model](args)
    model.to(device)
    utils.load_model(model, args.pretrained)

    if args.save_imgs:
        save_imgs_dir = os.path.join(args.output_dir, 'img')
        os.makedirs(save_imgs_dir, exist_ok=True)
    else:
        save_imgs_dir = None

    model.eval()
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)

    input_size = args.input_size
    size_dict = {
        256: (256, 256),
        512: (512, 512),
    }
    input_size = size_dict[input_size]

    img1 = Image.open(args.img_path_1).convert('RGB')
    img2 = Image.open(args.img_path_2).convert('RGB')
    img_list = [img1, img2]
    img_list = proc_image(img_list, F.resize, size=input_size)
    img_list = proc_image(img_list, F.to_tensor)
    img_list = proc_image(img_list, F.normalize, mean=mean, std=std)

    image = torch.cat(img_list, dim=0)

    with torch.no_grad():
        image = image.to(device).unsqueeze(0)
        output = model(image)
        if isinstance(output, OrderedDict):
            output = output['out']
        pred = output.argmax(1)
    
    img1 = np.array(img1).astype(float)
    img1 = cv2.resize(img1, input_size)
    img2 = np.array(img2).astype(float)
    img2 = cv2.resize(img2, input_size)
    pred_img = (pred.squeeze().cpu().numpy() * 255).astype(np.uint8)

    alpha = 0.5 * pred.squeeze().cpu().numpy()
    alpha = np.stack([alpha] * 3, axis=-1).astype(float)
    pred_img = np.stack([pred_img, np.zeros_like(pred_img), np.zeros_like(pred_img)], axis=-1).astype(float)

    foreground_1 = cv2.multiply(alpha, pred_img)
    background_1 = cv2.multiply(1 - alpha, img1)
    output_1 = cv2.add(foreground_1, background_1).astype(np.uint8)

    blank = np.zeros([512, 128, 3]).astype(np.uint8)
    full_vis = np.concatenate([output_1, blank, img2.astype(np.uint8)], axis=1)
    plt.imshow(full_vis)
    plt.show()
    return


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch change detection', add_help=add_help)
    parser.add_argument('--train-dataset', default='VL_CMU_CD', help='dataset name')
    parser.add_argument('--test-dataset', default='VL_CMU_CD', help='dataset name')
    parser.add_argument('--test-dataset2', default='', help='dataset name')
    parser.add_argument('--input-size', default=448, type=int, metavar='N',
                        help='the input-size of images')
    parser.add_argument('--randomflip', default=0.5, type=float, help='random flip input')
    parser.add_argument('--randomrotate', dest="randomrotate", action="store_true", help='random rotate input')
    parser.add_argument('--randomcrop', dest="randomcrop", action="store_true", help='random crop input')
    parser.add_argument('--data-cv', default=0, type=int, metavar='N',
                        help='the number of cross validation')

    parser.add_argument('--model', default='resnet18_mtf_msf_deeplabv3', help='model')
    parser.add_argument('--mtf', default='iade', help='choose branches to use')
    parser.add_argument('--msf', default=4, type=int, help='the number of MSF layers')
    
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=4, type=int)
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--loss', default='bi', type=str, help='the training loss')
    parser.add_argument('--loss-weight', action="store_true", help='add weight for loss')
    parser.add_argument('--opt', default='adam', type=str, help='the optimizer')
    parser.add_argument('--lr-scheduler', default='cosine', type=str, help='the lr scheduler')
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--warmup', dest="warmup", action="store_true", help='warmup the lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument("--pretrained", default='', help='pretrain checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval-every', default=1, type=int, metavar='N',
                        help='eval the model every n epoch')
    parser.add_argument("--test-only", dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument("--save-imgs", dest="save_imgs", action="store_true",
                        help="save the predicted mask")

    
    parser.add_argument("--save-local", dest="save_local", help="save logs to local", action="store_true")
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument("--img_path_1", required=True, type=str, help='name of first image file')
    parser.add_argument("--img_path_2", required=True, type=str, help='name of second image file')

    return parser


if __name__ == "__main__":
    #os.environ["TORCH_HOME"] = '/Pretrained'
    args = get_args_parser().parse_args()
    output_dir = 'output'
    save_path = "{}_{}_{}/{date:%Y-%m-%d_%H:%M:%S}".format(
        args.model, args.train_dataset, args.data_cv, date=datetime.datetime.now())
    args.output_dir = os.path.join(output_dir, save_path)

    main(args)
