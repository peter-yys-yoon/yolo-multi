from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import argparse

from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import cv2
from tqdm import tqdm

lower_ybound = 600
upper_ybound = 1800
normalized_labels = True


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # use center padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # add padding, pad format: (padding_left, padding_right, padding_top, padding_bottom)
    img = F.pad(img, pad, 'constant', value=pad_value)
    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def preprocess(_vis_img, img_size=416):
    # _vis_img = _vis_img[lower_ybound: upper_ybound, :, :]
    vis_img = cv2.cvtColor(_vis_img, cv2.COLOR_BGR2RGB)
    img = transforms.ToTensor()(vis_img)
    # pad to square resolution
    img, _ = pad_to_square(img, 0)  # note: can use mean_value
    # resize
    img = resize(img, img_size)

    # return img.unsqueeze(0)
    return img


def prepare_crop(frame, img_size=416):

    patch_torch_list = preprocess(frame, img_size).unsqueeze(0)

    return frame, patch_torch_list , [ (0,0) ]


def prepare_grid(valid_frame, img_size=416, step=800):
    """
    divide patch from valid-frame
    """
    # valid_frame = vis_frame[lower_ybound:upper_ybound, :, :]
    # cv2.imshow('33',raw_img)
    # print('Valid frame size:', valid_frame.shape)
    H, W = valid_frame.shape[:2]
    patch_torch_list = []
    patch_xy_list = []
    patch_vis_list = []
    for px in range(0, W - step + 1, 400):
        for py in range(0, H - step + 1, 400):

            pyy = lower_ybound + py
            patch = valid_frame[pyy:pyy + step, px:px + step, :]
            # print('patch extract (' ,xx, '~',xx + step,') (',yy,'~',yy+step,')', patch.shape)
            # print(xx, yy)
            # print(patch.shape)
            patch_vis_list.append(patch)
            patch_torch_list.append(preprocess(patch, img_size))
            patch_xy_list.append((px, pyy))

            # cv2.imshow('1', valid_frame)
            # cv2.imshow('org_patch', patch)
            # if cv2.waitKey(0) == ord('q'):
            #     quit()

    return patch_vis_list, torch.stack(patch_torch_list), patch_xy_list


# def person_detect(frame):

class my_detector:
    def __init__(self, opt, mode):
        print(opt)
        device = torch.device("cuda")
        os.makedirs("output", exist_ok=True)  # avoid existing dir cause OS error

        self.mode = mode
        # set up model
        self.model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
        if opt.weights_path.endswith(".weights"):
            self.model.load_darknet_weights(opt.weights_path)
        else:
            self.model.load_state_dict(torch.load(opt.weights_path))

        self.model.eval()  # fix BN, Dropot param
        self.classes = load_classes(opt.class_path)

    def detect(self, frame):

        if self.mode == 'grid':
            patch_vis_list, patch_list, patch_xy_list = prepare_grid(frame)
        else:
            patch_vis_list, patch_list, patch_xy_list = prepare_crop(frame)

        inputs = patch_list.cuda()

        with torch.no_grad():  # deactivate autograd
            detections = self.model(inputs)
            # input detections format: (x_ctr, y_ctr, w, h, objectness, cls_score)
            # return detections format: (x1, y1, x2, y2, objectness, cls_score, cls_label)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # bbox colors
        cmap = plt.get_cmap("tab20b")
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        # pdb.set_trace()


        frame_bbox_list = []
        for patch_img, patch_xy, detection in zip(patch_vis_list, patch_xy_list, detections):
            if detection is None:
                continue

            # rescale boxes to original image
            detections = rescale_boxes(detection, opt.img_size, patch_img.shape[:2])
            # unique labels
            unique_labels = detection[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for x1, y1, x2, y2, objectness, cls_score, cls_label in detections:
                print("label-%s %.2f %.2f %.2f %.2f %.2f" % (self.classes[int(cls_label)], cls_score, x1, y1, x2, y2))
                color = bbox_colors[int(np.where(unique_labels == int(cls_label))[0])]
                patch_x, patch_y = patch_xy
                x1 = x1 + patch_x
                y1 = y1 + patch_y
                frame_bbox_list.append([self.classes[int(cls_label)], cls_score, x1, y1, x2, y2])

        return frame_bbox_list


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples2")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights")
    # parser.add_argument("--weights_path", type=str, default="checkpoints/yolov3_epoch15.pth")
    parser.add_argument("--class_path", type=str, default="data/coco.names")
    parser.add_argument("--conf_thres", type=float, default=0.8)
    parser.add_argument("--nms_thres", type=int, default=0.4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--n_cpu", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=416)
    parser.add_argument("--checkpoint_model", type=str, help="pytorch model")
    opt = parser.parse_args()

    imgs = []
    img_detections = []
    path = '/home/peter/workspace/dataset/lotte20/6/camera1/high/HighLight-6adults-backstroke-drowningB/HighLight-6adults-backstroke-drowningB.mp4'
    # path = '/home/peter/Desktop/ldcc_data_tmp/6/camera1/high/HighLight-6adults-backstroke-drowningB/HighLight-6adults-backstroke-drowningB.mp4'
    cap = cv2.VideoCapture(path)
    name_desc = tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    framecount = 0
    while 1:
        name_desc.update(1)
        ret, vis_frame = cap.read()
        if ret is False:
            break
        # print(framecount)
        # framecount +=1
        # cv2.imshow('first' ,cv2.resize(vis_frame,(int(4000/3),1000)))
        # if cv2.waitKey(0) == ord('q'):
        #     quit()

        # cv2.imshow('pr', patch_img)
        cv2.imshow('or', cv2.resize(valid_frame, (int(valid_frame.shape[1] / 3), int(valid_frame.shape[0] / 3))))
        if cv2.waitKey(0) == ord('q'):
            quit()
