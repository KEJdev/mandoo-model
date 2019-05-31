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
<<<<<<< HEAD
    
=======

def convert_to_query_db_data_fixed_window(img_list, label_list, input_size, num_query, max_ref_count):
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
            img = image_load(img, img_size=input_size[:2])
            queries_img.append(img)
            label_visit.append(label)
        elif label in label_visit:
            if (label in label_reference_cnt.keys()) and label_reference_cnt[label] > max_ref_count:
                continue
            else:
                label_reference_cnt[label] = label_reference_cnt.get(label, 0) + 1
            references.append(key)
            img = image_load(img, img_size=input_size[:2])
            reference_img.append(img)
    return queries, references, queries_img, reference_img

def convert_to_query_db_data(img_list, label_list, input_size, num_classes, max_ref_count, debug):
    """ load image with labels from filename"""
    label_reference_cnt = {}
    label_visit = []
    queries = []
    queries_img = []
    references = []
    reference_img = []
    if not debug: 
        labels = range(0, num_classes, 7)
    else:
        labels = range(0, 15)
    print("query_num")
    print(len(labels))
    for i, (img, label) in enumerate(zip(img_list, label_list)):
        key = "/" + str(label) + "@" + str(i) + ".jpg"
        if label not in label_visit and label in labels:
            queries.append(key)
            img = image_load(img, img_size=input_size[:2])
            queries_img.append(img)
            label_visit.append(label)
        elif label in label_visit:
            if (label in label_reference_cnt.keys()) and label_reference_cnt[label] > max_ref_count: 
                continue
            else:
                label_reference_cnt[label] = label_reference_cnt.get(label, 0) + 1
            references.append(key)
            img = image_load(img, img_size=input_size[:2])
            reference_img.append(img)
    return queries, references, queries_img, reference_img

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)
      name_list = name.split('/')
      if name_list[0] == 'Nasnet_block':
          name = '/'.join(name_list[1:])
          name += '/'
    name_to_variable[name] = var

  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


