from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import argparse
import tqdm
import pdb
import torch
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import datetime

def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        collate_fn=dataset.collate_fn,
    )

    Tensor = torch.cuda.FloatTensor

    labels = []
    sample_metrics = [] # list of tuples (TP, confs, pred)

    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # extract class labels
        labels += targets[:, 1].tolist() # format: (sample_index, class, x_ctr, y_ctr, w, h)
        # get normalized (xmin, ymin, xmax, ymax)
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        # get w.r. 416x416 (xmin, ymin, xmax, ymax)
        targets[:, 2:] *= img_size

        imgs = imgs.cuda()
        # imgs = imgs.to('cuda')
        # print('img shape', imgs.shape)
        with torch.no_grad():
            # outputs size: [n, 10647, 85]
            outputs = model(imgs)
            # print('model done--------------------')
            outputs = non_max_suppression(outputs, conf_thres, nms_thres)
            # print('nms done--------------------')
            # nms-outputs is a list, elment is tensor
            # outputs[0]: torch.size([valid_box_number, 7]) 7 is (x1, y1, x2, y2, conf, class_score, class_pred)

        # sample_metrics is a list, elem format (true_positives, pred_scores, pred_labels)
        # length of sample_metrics: batch_size + batch_size + ....


        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # sample_metrics = sample_metrics.cpu()
    # concat sample statistics
    # true_positives, pred_scores, pred_labels = [np.concatenate(x.cpu(), 0) for x in list(zip(*sample_metrics))]
    # pdb.set_trace()
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class




def video_class_test():
    path = '/home/peter/workspace/dataset/lotte20/6/camera1/high/HighLight-6adults-backstroke-drowningA/HighLight-6adults-backstroke-drowningA.mp4'
    dataset = VideoDataset(path, img_size=416, augment=False, multiscale=False)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        collate_fn=dataset.collate_fn,
    )

    Tensor = torch.cuda.FloatTensor

    labels = []
    sample_metrics = []  # list of tuples (TP, confs, pred)

    for batch_i, (vis_images, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # extract class labels
        print('batch', batch_i)
        print('imgs:', vis_images)
        print()




if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=24, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/tang.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.001, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
    )


    with open('acclog.txt','a') as f:
        print(precision, recall, AP, f1, ap_class)
        #pdb.set_trace()
        msg = ' '.join([opt.weights_path,'conf:', str(opt.conf_thres),'nms:',str(opt.nms_thres),'iou:',str(opt.iou_thres),'prec:', f'{precision[0]:.3f}','recall:',f'{recall[0]:.3f}','AP:',f'{AP[0]:.3f}','F1:',f'{f1[0]:.3f}'])
        f.write(msg+'\n')
        print(msg)
        #f.write(' '.join[opt.weights_path, str(opt.conf_thres),str(opt.iou_thres), f'{precision[0]:3.f}',f'{recall[0]:.3}',f'{AP[0]:.3f}','\n'])
    #print(precision, recall, AP, f1, ap_class)

    #print("Average Precisions:")
    #for i, c in enumerate(ap_class):
    #    print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

    #print(f"mAP: {AP.mean()}")
