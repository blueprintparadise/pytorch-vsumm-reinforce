#!/usr/bin/python
# -*- coding:utf-8 -*-
from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli

from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from rewards import compute_reward
import vsum_tools
import cv2

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options

parser.add_argument('-s', '--split', type=str, help="path to split file (required)")
parser.add_argument('--split-id', type=int, default=0, help="split index (default: 0)")
# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
# Model options
parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
#输入视频的feature H5file
parser.add_argument('-d', '--dataset', type=str, help="path to h5 dataset (required)")
#打分，存储h5
parser.add_argument('--makescore', action='store_true', help=" ")
parser.add_argument('--model', type=str, default='', help="path to model file")
parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
#视频摘要生成
parser.add_argument('--summary', action='store_true', help=" ")
parser.add_argument('--frm-dir', help="'frames/'")
parser.add_argument('--makedatasets', action='store_true', help=" ")
parser.add_argument('--video-dir', help="'video/'")
parser.add_argument('--save-name', default='summary.mp4',help="'generate video '")
parser.add_argument('--fps', type=int, default=30, help="frames per second")
parser.add_argument('--width', type=int, default=640, help="frame width")
parser.add_argument('--height', type=int, default=480, help="frame height")

parser.add_argument('--train-data', action='store_true', help="")
'''
#生成数据
python video_forward.py --makedatasets --dataset data_our/data_h5/data1.h5  --video-dir data_video/data1/ --frm-dir data_our/frames   
#输出摘要
python3 video_forward.py --makescore --model log/summe-split0/model_epoch1000.pth.tar --gpu 0 --dataset data_our/data_h5/data1.h5 --save-dir logs/videolog/ \
--summary  --frm-dir data_our/frames    
#生成数据 数据源
python3 video_forward.py --makedatasets --dataset-dir data_our/data_h5/  --video-dir data_video/ \
--makescore --model log/summe-split0/model_epoch1000.pth.tar --gpu 0 --dataset utils/train.h5 --save-dir logs/videolog/ \
--summary  --frm-dir data_our/frames   
#直接输出摘要 模型；数据集H5；输出路径；   视频帧存储路径；
python3 video_forward.py --makescore --model log/summe-split0/model_epoch1000.pth.tar --gpu 0 --dataset utils/train.h5 --save-dir logs/videolog/ \
--summary  --frm-dir utils/frames   


python3 video_forward.py --makedatasets --dataset-dir videoframes/  --video-dir data_video/
python3 video_forward.py --summary --frm-dir videoframes/ --save-dir logs/videolog/
python3 video_forward.py --summary --dataset utils/train.h5 --frm-dir utils/frames --save-dir logs/videolog/ -s datasets/traintest.json  --gpu 0 --model log/summe-split0/model_epoch1000.pth.tar
'''
args = parser.parse_args()

torch.manual_seed(args.seed)
#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False
use_gpu = False
def main():
    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset))
    dataset = h5py.File(args.dataset, 'r')
    num_videos = len(dataset.keys())
    test_keys = []
    if args.split:
        splits = read_json(args.split)
        assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
        split = splits[args.split_id]
        test_keys = split['test_keys']
    else:
        for key in dataset.keys():
            test_keys.append(key)
    print(test_keys)

    print("# total videos {}. # test videos {}".format(num_videos,  len(test_keys)))

    print("Initialize model")
    model = DSN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
    print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #if args.stepsize > 0:
    #    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)

    if args.model:
        print("Loading checkpoint from '{}'".format(args.model))
        checkpoint = torch.load(args.model)
        model.load_state_dict(checkpoint)
    else:
        start_epoch = 0

    if use_gpu:
        model = nn.DataParallel(model).cuda()

    print("Evaluate")
    evaluate(model, dataset, test_keys, use_gpu)
    print("Summary")
    if args.summary:
        video2summary(os.path.join(args.save_dir,'result.h5'),args.frm_dir,args.save_dir)####


####
# 输出score图以及h5文件
def evaluate(model, dataset, test_keys, use_gpu):
    print("==> Test")
    with torch.no_grad():
        model.eval()
        fms = []

        table = [["No.", "Video", "F-score"]]
        if not os.path.isdir(args.save_dir):
            os.mkdir(args.save_dir)

        h5_res = h5py.File(os.path.join(args.save_dir,'result.h5'), 'w')

        for key_idx, key in enumerate(test_keys):
            seq = dataset[key]['features'][...]
            seq = torch.from_numpy(seq).unsqueeze(0)
            if use_gpu: seq = seq.cuda()
            probs = model(seq)
            probs = probs.data.cpu().squeeze().numpy()

            cps = dataset[key]['change_points'][...]
            num_frames = dataset[key]['n_frames'][()]
            nfps = dataset[key]['n_frame_per_seg'][...].tolist()
            positions = dataset[key]['picks'][...]
            video_name = dataset[key]['video_name'][()]
            video_dir = dataset[key]['video_dir'][()]
            fps = dataset[key]['fps'][()]
            #print(cps)
            #print(nfps)
            sum = 0
            for i in range(len(nfps)):
                sum += nfps[i]
            #print(sum)
            #print(positions)
            #video_name = 'train'
            machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
            #print(video_name)
            #print(":")
            print(machine_summary.shape)
            h5_res.create_dataset(key + '/score', data=probs)
            h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
            h5_res.create_dataset(key + '/video_name', data=video_name)
            h5_res.create_dataset(key + '/fps', data=fps)
            h5_res.create_dataset(key + '/video_dir', data=video_dir)

    h5_res.close()

def frm2video(frm_dir, summary, vid_writer,video_dir):
    video_capture = cv2.VideoCapture(video_dir)

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    n_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    cnt = 0
    for idx, val in enumerate(summary):
        success, frame = video_capture.read()

        if val == 1:
            frm = cv2.resize(frame, (args.width, args.height))
            vid_writer.write(frm)
        cnt += 1

def video2summary(h5_dir,frm_dir,output_dir):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)

    h5_res = h5py.File(h5_dir, 'r')

    ####遍历生成###
    print(list(h5_res.keys()))
    for idx1 in range(len(list(h5_res.keys()))):
        #print(idx1)
        #print(":")
        key = list(h5_res.keys())[idx1]
        summary = h5_res[key]['machine_summary'][...]
        video_name = h5_res[key]['video_name'][()]
        video_dir = h5_res[key]['video_dir'][()]
        #video_name = "train"
        video_name = str(video_name, encoding="utf-8")
        video_dir = str(video_dir, encoding="utf-8")
        fps = h5_res[key]['fps'][()]
        #print(video_name)
        #print(osp.join(args.save_dir,video_name, args.save_name))
        if not os.path.isdir(osp.join(output_dir, video_name)):
            os.mkdir(osp.join(output_dir, video_name))
        vid_writer = cv2.VideoWriter(
            #osp.join(args.save_dir, dict1[key], args.save_name),
            osp.join(output_dir,video_name, "summary.mp4"),
            cv2.VideoWriter_fourcc(*'MP4V'),
            fps,#args.fps,
            (args.width, args.height),
        )
        frm2video(frm_dir, summary, vid_writer,video_dir)
        vid_writer.release()
    h5_res.close()



if __name__ == '__main__':
    if args.makedatasets:
        from utils.generate_dataset import Generate_Dataset
        video = args.video_dir
        output = args.dataset
        gen = Generate_Dataset(video,output,args.frm_dir,args.train_data)
        gen.generate_dataset()
        gen.h5_file.close()
    if args.makescore:
        main()
