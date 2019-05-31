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
import random
from train_utils import l2_normalize
from imgaug import augmenters as iaa
import imgaug as ia

def image_load(img_path, img_size):
    img = cv2.imread(img_path, 1)
    height, width, channel = img.shape
    square_side = min(height, width)
    top_height = int((height - square_side) / 2)
    left_width = int((width - square_side) / 2)
    img = img[top_height:top_height + square_side,
             left_width:left_width + square_side]
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, img_size)
    return img

def train_data_loader(data_path, output_path):
    label_list = []
    img_list = []
    label_idx = 0

    for root, dirs, files in os.walk(data_path):
        if not files:
            continue
        for filename in files:
            img_path = os.path.join(root, filename)
            label_list.append(label_idx)
            img_list.append(img_path)
        label_idx += 1

    # write output file for caching
    with open(output_path[0], 'wb') as img_f:
        pickle.dump(img_list, img_f)
    with open(output_path[1], 'wb') as label_f:
        pickle.dump(label_list, label_f)

def image_generator(img_paths):
    img_size = (224, 224)
    for img_path in img_paths:
        img = image_load(img_path, img_size)
        img = np.asarray(img).astype('float32')
        yield img

def query_expand_generator(img_paths):
    img_size = (224, 224)
    for img_path in img_paths:
        img = image_load(img_path, img_size)
        img = np.asarray(img).astype('float32')
        seq = iaa.Sequential(iaa.Noop())
        fliplr_seq = iaa.Sequential(iaa.Fliplr(1.0))
        flipud_seq = iaa.Sequential(iaa.Flipud(1.0))
        rotate_seq = iaa.Sequential(iaa.Affine(rotate=(-45.0, 45.0)))
        seq_list = [seq, fliplr_seq, flipud_seq, rotate_seq]
        imgs = []
        for seq in seq_list:
            imgs.append(seq.augment_image(img))
        yield imgs

def generator(
    train_dataset_path, 
    num_classes=1383, 
    input_shape=(224, 224)):
    label_number = 0
    for outputs in os.walk(train_dataset_path):
        root, _, files = outputs
        if not files:
            continue
        for filename in files:
            img_path = os.path.join(root, filename)
            try:
                img = image_load(img_path, img_size=input_shape[:2])
            except:
                continue
            y_cate = tf.keras.utils.to_categorical(
                label_number, num_classes=num_classes)
            yield (img, y_cate)
        label_number += 1
    
def convert_to_query_db_data_for_generator(
    img_list, 
    label_list, 
    input_size, 
    num_query, 
    max_ref_count):
    """ load image with labels from filename"""
    label_reference_cnt = {}
    label_visit = []
    queries = []
    queries_img = []
    references = []
    reference_img = []
    used_datapath = []
    for i, (img, label) in enumerate(zip(img_list, label_list)):
        key = "/" + str(label) + "@" + str(i) + ".jpg"
        if label not in label_visit and len(label_visit) < num_query:
            queries.append(key)
            queries_img.append(img)
            label_visit.append(label)
        elif label in label_visit:
            if (label in label_reference_cnt.keys()) and label_reference_cnt[label] > max_ref_count:
                continue
            else:
                label_reference_cnt[label] = label_reference_cnt.get(label, 0) + 1
            references.append(key)
            reference_img.append(img)
    return queries, references, queries_img, reference_img
    