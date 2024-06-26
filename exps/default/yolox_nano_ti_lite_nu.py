#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

import torch.nn as nn

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # ---------------- model config ---------------- #
        self.num_classes = 6
        self.depth = 0.33
        self.width = 0.25
        self.input_size = (320, 320)
        self.random_size = (10, 20)
        self.mosaic_scale = (0.5, 1.5)
        self.mosaic_prob = 0.5
        self.enable_mixup = False
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.act = "relu"

        # Define yourself dataset path
        self.data_dir = "datasets/medicinev2_coco"
        self.train_ann = "medicinev2_train.json"
        self.val_ann = "medicinev2_val.json"

        # --------------  training config --------------------- #
        self.warmup_epochs = 5
        self.max_epoch = 200
        # -----------------  testing config ------------------ #
        self.test_size = (320, 320)

    def get_model(self, sublinear=False):

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03
        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO model use depthwise = True, which is main difference.
            #backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, depthwise=False, conv_focus=True, split_max_pool_kernel=True)
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, depthwise=True, conv_focus=True, split_max_pool_kernel=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act, depthwise=False)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
