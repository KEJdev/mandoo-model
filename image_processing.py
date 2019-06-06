from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import pickle
import numpy as np
import collections
import tensorflow as tf
import re
from random import *
from data_loader import image_load
from imgaug import augmenters as iaa
import imgaug as ia

# data preprocess

def preprocess(queries, references):
     query_img = []
     reference_img = []
     img_size = (224, 224)
     for img_path in queries:
         img = image_load(img_path, img_size)
         query_img.append(img)
     for img_path in references:
         img = image_load(img_path, img_size)
         reference_img.append(img)
     return queries, query_img, references, reference_img

def get_aug_config(config): 
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    settings = []
    if config.crop:
        settings.append(sometimes(iaa.Crop(px=(0, 16))))
    if config.fliplr:
        settings.append(sometimes(iaa.Fliplr(0.5)))
    if config.flipud:
        settings.append(sometimes(iaa.Flipud(0.5)))
    if config.gausian:
        settings.append(sometimes(iaa.GaussianBlur(sigma=(0, 3.0))))
    if config.dropout:
        settings.append(sometimes(iaa.Dropout(0.02, name="Dropout")))
    if config.noise:
        settings.append(sometimes(iaa.AdditiveGaussianNoise(scale=0.01*255, name="MyLittleNoise")))
    if config.rotate:
        settings.append(sometimes(iaa.Affine(
            # scale images to 80-120% of their size, individually per axis
            #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            # translate by -20 to +20 percent (per axis)
            #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-45, 45), # rotate by -45 to +45 degrees
            #shear=(-16, 16), # shear by -16 to +16 degrees
            #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
            #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
            # use any of scikit-image's warping modes (see 2nd image from the top for examples))))
            mode=ia.ALL
        )))
    return settings
