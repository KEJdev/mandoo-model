from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import argparse
import time
import pickle
import sys
import math

import nsml
from nsml import DATASET_PATH
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

import numpy as np
import tensorflow as tf
from image_processing import preprocess, get_aug_config
from data_loader import get_assignment_map_from_checkpoint,\
    get_balanced_dual_dataset, get_dual_dataset, image_load, train_data_loader, \
    convert_to_query_db_data, convert_to_query_db_data_fixed_window, \
    convert_to_query_db_data_for_generator
from measure import evaluate_mAP, evaluate_rank
from inference import get_feature
from train_utils import l2_normalize
from loss import batch_hard_triplet_loss
from model.delf_model import *
from imgaug import augmenters as iaa
import imgaug as ia

from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io
from scipy.spatial import cKDTree

local_infer = None

def bind_model(sess):
    global local_infer

    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        saver = tf.train.Saver()
        saver.save(sess, os.path.join(dir_name, 'model'))

    def load(file_path):
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(file_path)
        if ckpt and ckpt.model_checkpoint_path:
            checkpoint = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(file_path, checkpoint))
        else:
            raise NotImplementedError('No checkpoint!')
        print('model loaded :' + file_path)

    def infer(queries, references, _query_img=None, _reference_img=None, batch_size=128):

        # load, and process images
        if _query_img is None:
            # not debug
            test_path = DATASET_PATH + '/test/test_data'
            db = [os.path.join(test_path, 'reference', path) for path in os.listdir(os.path.join(test_path, 'reference'))]
            queries = [v.split('/')[-1].split('.')[0] for v in queries]
            db = [v.split('/')[-1].split('.')[0] for v in db]
            queries.sort()
            db.sort()
            queries_full_paths = list(map(lambda x: '/data/ir_ph2/test/test_data/query/' + x + '.jpg', queries))
            db_full_path = list(map(lambda x: '/data/ir_ph2/test/test_data/reference/' + x + '.jpg', db))
            _, query_vecs, _, reference_vecs = get_feature(queries_full_paths, db_full_path, sess, batch_size)

        else:
            # debug
            _, query_vecs, _, reference_vecs = get_feature(_query_img, _reference_img, sess, batch_size)
            db = references

        kd_tree = cKDTree(query_vecs)
        _, indices = kd_tree.query(reference_vecs, distance_upper_bound = 0.8)

        retrieval_results = {}

        query_vecs = l2_normalize(query_vecs)
        reference_vecs = l2_normalize(reference_vecs)
        print("indices == {}".format(indices))

        def get_ranked_list(query, reference_vecs):
            sim_matrix = np.dot(query, reference_vecs.T)
            sub_indices = np.argsort(sim_matrix)
            sub_indices = np.flip(sub_indices)
            ranked_list = [db[k] for k in sub_indices]
            return ranked_list
        
        queries_name = {}
        for idx, query_name in enumerate(queries):
            queries_name[idx] = query_name

        for i, query in enumerate(query_vecs):
            ref_indices = np.argwhere(indices == i)
            if len(ref_indices) == 0:
                ranked_list = get_ranked_list(query, reference_vecs)
            else:
                clustered_ref = reference_vecs[ref_indices]
                left_over_indices = np.argwhere(indices != i)
                left_over_ref = reference_vecs[left_over_indices]
                clustered_ranked_list = get_ranked_list(query, clustered_ref)
                left_over_ranked_list = get_ranked_list(query, left_over_ref)
                ranked_list = clustered_ref + left_over_ref
            retrieval_results[queries_name[i]] = ranked_list
        return list(zip(range(len(retrieval_results)), retrieval_results.items()))
